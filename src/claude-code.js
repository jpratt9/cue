// Claude Code provider — drives the locally installed `claude` CLI in headless
// mode instead of calling api.anthropic.com with an API key, so cue runs on
// whatever auth the user's Claude Code session already has.
//
// Uses the documented headless interface (`--print` with
// `--input-format/--output-format stream-json`), which takes the same message
// objects the Anthropic SDK does — images included — and emits token deltas we
// can forward straight to cue's onToken. Deliberately NOT the OAuth token out
// of the login keychain: reusing that credential against the raw API means
// spoofing Claude Code's identity headers, and accounts doing it get banned.
//
// WHY A STANDBY POOL. cue calls stream() with one self-contained user turn per
// request (main.js renders the conversation into the prompt text rather than
// passing a turn list), so a request can never continue the previous request's
// session — every one needs a conversation with no history. The naive
// implementation therefore spawns a process per request and pays Node+CLI boot
// on every hotkey press. Measured on haiku:
//
//   cold spawn, then send        ~1870ms
//   already-spawned, then send   ~1280ms   <- boot paid off the critical path
//
// So each key keeps one spare process spawned and waiting. A request takes the
// spare, uses it, kills it (its session now has history), and a replacement is
// spawned in the background for the next one. Note the CLI stays silent until
// it receives input — it emits no `init` until the first message — so there is
// nothing to wait for after spawning; Node boot simply completes off-path.
//
// Every process is isolated from the user's own Claude Code config
// (--setting-sources "" --strict-mcp-config, a scratch cwd, no tools): cue
// sends its own system prompt and wants plain-text completions, not the user's
// CLAUDE.md, skills, hooks or MCP servers loaded on a hotkey.

const { spawn } = require('child_process');
const crypto = require('crypto');
const fs = require('fs');
const os = require('os');
const path = require('path');

const CLAUDE_CODE_PROVIDER = 'claudecode';

// Aliases, not pinned ids: the CLI resolves each to the current release, so
// these keep tracking the newest model without another model-id fix like the
// retired-id migrations in llm.js.
//
// Verified against this machine's CLI (2.1.283) by running a real completion
// per id — `init` echoes whatever string you pass, including nonsense, so only
// a successful result proves a model exists:
//   opus   -> claude-opus-5-5              OK
//   sonnet -> claude-sonnet-5              OK
//   haiku  -> claude-haiku-4-5-20251001    OK
//   claude-sonnet-5-5 / claude-haiku-5-5 / claude-haiku-5   all rejected
// There is no Sonnet 5.5 and no Haiku 5.x, so `haiku` already IS the newest
// Haiku — these aliases are not pinned to an old generation, and they pick up
// a Haiku 5 / Sonnet 5.5 automatically on the day one ships.
const DEFAULT_MODELS = { fast: 'haiku', smart: 'sonnet' };

// How long a spare process is kept warm after the last request. Past this cue
// is probably idle, and holding a spawned Claude Code process (plus its slot
// server-side) to save 600ms on a request that may never come is the wrong
// trade.
const STANDBY_IDLE_MS = 5 * 60 * 1000;

// A GUI Electron app launched from Finder/Dock inherits a bare PATH
// (/usr/bin:/bin:/usr/sbin:/sbin) — not the shell PATH that has `claude` on
// it — so resolving the binary by name alone fails in a packaged build even
// though it works from `npm start` in a terminal. Check the install locations
// the official installers use before falling back to PATH.
const CANDIDATE_PATHS = [
  path.join(os.homedir(), '.local', 'bin', 'claude'),
  path.join(os.homedir(), '.claude', 'local', 'claude'),
  '/opt/homebrew/bin/claude',
  '/usr/local/bin/claude'
];

function resolveCli(configuredPath) {
  const explicit = (configuredPath || '').trim();
  if (explicit) return explicit;
  for (const candidate of CANDIDATE_PATHS) {
    try {
      fs.accessSync(candidate, fs.constants.X_OK);
      return candidate;
    } catch {}
  }
  return 'claude';
}

// Claude Code resolves CLAUDE.md and file tools against its working directory.
// Point it at an empty scratch dir so a hotkey pressed while cue happens to be
// over a repo cannot pull that repo's instructions into the prompt.
function scratchCwd() {
  const dir = path.join(os.tmpdir(), 'cue-claude-code');
  try { fs.mkdirSync(dir, { recursive: true }); } catch {}
  return dir;
}

// The CLI has no max_tokens flag, so the length cap that other providers pass
// as a parameter has to be stated in the prompt instead.
function systemPromptWithBudget(system, maxTokens) {
  const budget = `Keep the whole reply under roughly ${Math.max(1, Math.round(maxTokens * 0.7))} words. Reply with the answer only — no preamble, no tool use, no questions back.`;
  return system ? `${system}\n\n${budget}` : budget;
}

function encodeTurn(turn, image) {
  if (image && turn.role === 'user') {
    return {
      type: 'user',
      message: {
        role: 'user',
        content: [
          { type: 'image', source: { type: 'base64', media_type: image.mime, data: image.b64 } },
          { type: 'text', text: turn.text }
        ]
      }
    };
  }
  return { type: turn.role, message: { role: turn.role, content: turn.text } };
}

function friendlyError(stderr, code, cliPath) {
  const text = String(stderr || '');
  if (/ENOENT|command not found|no such file/i.test(text) || code === 127) {
    return `Claude Code CLI not found at "${cliPath}". Install it, or set the path in Settings → Provider.`;
  }
  if (/not logged in|\/login|authentication_error|invalid.*api.?key|unauthorized|401/i.test(text)) {
    return 'Claude Code is not logged in. Run `claude` in a terminal, finish /login, then retry.';
  }
  if (/usage limit|rate.?limit|429/i.test(text)) {
    return 'Claude Code session hit its usage limit. Wait for the window to reset or switch providers in Settings.';
  }
  const firstLine = text.split('\n').map(l => l.trim()).filter(Boolean)[0];
  return firstLine ? `Claude Code CLI failed: ${firstLine}` : `Claude Code CLI exited with code ${code}.`;
}

// One Claude Code process, good for exactly one request. Spawned ahead of time
// so Node boot is already done when a request claims it.
class Session {
  constructor({ bin, model, system, spawnImpl = spawn }) {
    this.bin = bin;
    this.stderr = '';
    this.buffer = '';
    this.pending = null;
    this.exited = false;
    this.exitError = null;

    this.child = spawnImpl(bin, [
      '--print',
      '--model', model,
      '--input-format', 'stream-json',
      '--output-format', 'stream-json',
      '--verbose',
      '--include-partial-messages',
      '--system-prompt', system,
      '--exclude-dynamic-system-prompt-sections',
      // Isolation: ignore the user's own settings, skills, hooks and MCP servers.
      '--setting-sources', '',
      '--strict-mcp-config',
      // cue wants one text completion; nothing here should touch the
      // filesystem or network on its own.
      '--disallowed-tools', 'Bash,Edit,Write,Read,Glob,Grep,WebFetch,WebSearch,Task,NotebookEdit,TodoWrite'
    ], { cwd: scratchCwd(), stdio: ['pipe', 'pipe', 'pipe'] });

    // A spare sitting unused must never take the app down on a broken pipe.
    this.child.stdin.on('error', () => {});
    this.child.stdout.setEncoding('utf8');
    this.child.stdout.on('data', (chunk) => this.onStdout(chunk));
    this.child.stderr.setEncoding('utf8');
    this.child.stderr.on('data', (chunk) => { this.stderr += chunk; });
    this.child.on('error', (error) => this.onExit(friendlyError(error.message, error.code, bin)));
    this.child.on('close', (code) => {
      this.onExit(code === 0 ? null : friendlyError(this.stderr, code, bin));
    });
  }

  onExit(message) {
    this.exited = true;
    this.exitError = message;
    const req = this.pending;
    this.pending = null;
    if (req) req.reject(new Error(message || 'Claude Code exited before it answered.'));
  }

  onStdout(chunk) {
    this.buffer += chunk;
    let newline;
    while ((newline = this.buffer.indexOf('\n')) !== -1) {
      const line = this.buffer.slice(0, newline).trim();
      this.buffer = this.buffer.slice(newline + 1);
      if (!line) continue;
      let event;
      try { event = JSON.parse(line); } catch { continue; }
      const req = this.pending;
      if (!req) continue;
      // Partial-message deltas are the token stream; `result` closes the turn
      // and is the fallback when no partials arrived.
      if (event.type === 'stream_event') {
        const delta = event.event && event.event.delta;
        if (delta && delta.type === 'text_delta' && delta.text) {
          req.text += delta.text;
          req.onToken(delta.text);
        }
      } else if (event.type === 'result') {
        this.pending = null;
        if (event.is_error) {
          req.reject(new Error(friendlyError(event.result || event.error, 0, this.bin)));
          return;
        }
        if (!req.text && typeof event.result === 'string') {
          req.text = event.result;
          if (req.text) req.onToken(req.text);
        }
        req.resolve(req.text);
      }
    }
  }

  run(turns, image, onToken) {
    if (this.exited) return Promise.reject(new Error(this.exitError || 'Claude Code exited before it answered.'));
    return new Promise((resolve, reject) => {
      this.pending = { onToken, resolve, reject, text: '' };
      // Only the final turn can carry the screenshot.
      turns.forEach((turn, i) => {
        const withImage = i === turns.length - 1 ? image : null;
        this.child.stdin.write(JSON.stringify(encodeTurn(turn, withImage)) + '\n');
      });
    });
  }

  kill() {
    // Idempotent: an aborted request kills the process, and the finally block
    // that retires it then calls this again.
    if (this.exited) return;
    this.exited = true;
    try { this.child.stdin.end(); } catch {}
    try { this.child.kill('SIGTERM'); } catch {}
  }
}

// One spare process per (binary, model, system prompt) — the three things fixed
// at spawn time. cue uses a different system prompt per feature, so asking a
// question and solving a screenshot keep separate spares instead of
// invalidating each other's.
const standbys = new Map();

function poolKey(bin, model, system) {
  return `${bin} ${model} ${crypto.createHash('sha1').update(system).digest('hex')}`;
}

function takeStandby(key) {
  const entry = standbys.get(key);
  if (!entry) return null;
  if (entry.timer) clearTimeout(entry.timer);
  standbys.delete(key);
  // A spare that died while waiting (usage limit, logout, machine sleep) is
  // useless; the caller spawns a fresh one instead.
  return entry.session.exited ? null : entry.session;
}

function putStandby(key, spec) {
  if (standbys.has(key)) return;
  let session;
  try {
    session = new Session(spec);
  } catch {
    return; // Spawning a spare is best-effort; the next request retries.
  }
  const timer = setTimeout(() => {
    const entry = standbys.get(key);
    if (entry && entry.session === session) { standbys.delete(key); session.kill(); }
  }, STANDBY_IDLE_MS);
  if (timer.unref) timer.unref();
  standbys.set(key, { session, timer });
}

async function streamClaudeCode({ model, system, turns, image, maxTokens, onToken, cliPath, signal, spawnImpl }) {
  const bin = resolveCli(cliPath);
  const prompt = systemPromptWithBudget(system, maxTokens);
  const spec = { bin, model: model || DEFAULT_MODELS.fast, system: prompt, spawnImpl };
  const key = poolKey(spec.bin, spec.model, spec.system);

  let session = takeStandby(key);
  if (!session) {
    try {
      session = new Session(spec);
    } catch (error) {
      throw new Error(friendlyError(error.message, error.code, bin));
    }
  }

  const abort = () => session.kill();
  if (signal) {
    if (signal.aborted) { abort(); throw new Error('aborted'); }
    signal.addEventListener('abort', abort, { once: true });
  }

  try {
    return await session.run(turns, image, onToken);
  } finally {
    if (signal) signal.removeEventListener('abort', abort);
    // The session now holds this request's history, so it cannot serve another.
    session.kill();
    putStandby(key, spec);
  }
}

function shutdown() {
  for (const [key, entry] of standbys) {
    if (entry.timer) clearTimeout(entry.timer);
    entry.session.kill();
    standbys.delete(key);
  }
}

module.exports = { CLAUDE_CODE_PROVIDER, DEFAULT_MODELS, streamClaudeCode, resolveCli, shutdown };
