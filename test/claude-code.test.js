// Every test here fakes the `claude` CLI through spawnImpl — nothing in this
// file starts a real process or reaches Anthropic.
const assert = require('node:assert/strict');
const { EventEmitter } = require('node:events');
const { PassThrough } = require('node:stream');
const test = require('node:test');
const claudeCode = require('../src/claude-code');
const { streamClaudeCode, shutdown, resolveCli, DEFAULT_MODELS } = claudeCode;

// Stands in for a spawned `claude --print --output-format stream-json` process:
// collects what cue writes to stdin and replays scripted stream-json events.
class FakeCli extends EventEmitter {
  constructor() {
    super();
    this.stdout = new PassThrough();
    this.stderr = new PassThrough();
    this.stdin = new PassThrough();
    this.written = [];
    this.killSignals = [];
    this.exitCode = null;
    this.stdin.on('data', (chunk) => {
      for (const line of String(chunk).split('\n')) {
        if (line.trim()) this.written.push(JSON.parse(line));
      }
      this.emit('wrote');
    });
  }

  emitEvent(event) { this.stdout.write(JSON.stringify(event) + '\n'); }

  // The real CLI streams text as partial-message deltas, then closes the turn
  // with a `result`.
  respond(text, { partial = true } = {}) {
    if (partial) {
      // Chunk on word boundaries but keep the separators, the way a real token
      // stream arrives — concatenating the deltas must rebuild `text` exactly.
      for (const piece of text.match(/\S+\s*/g) || []) {
        this.emitEvent({ type: 'stream_event', event: { delta: { type: 'text_delta', text: piece } } });
      }
    }
    this.emitEvent({ type: 'result', subtype: 'success', is_error: false, result: text });
  }

  fail(message) {
    this.emitEvent({ type: 'result', subtype: 'error', is_error: true, result: message });
  }

  kill(signal) {
    this.killSignals.push(signal);
    this.exitCode = 0;
    queueMicrotask(() => this.emit('close', 0));
    return true;
  }
}

// Hands out FakeCli instances and records the argv each was spawned with.
function fakeSpawner() {
  const children = [];
  const spawner = (bin, args, options) => {
    const child = new FakeCli();
    child.spawnArgs = { bin, args, options };
    children.push(child);
    return child;
  };
  // Waits for the Nth process to exist, since standbys are spawned off-path.
  spawner.nth = async (index) => {
    for (let i = 0; i < 200 && children.length <= index; i++) await new Promise((r) => setTimeout(r, 5));
    assert.ok(children.length > index, `expected a process at index ${index}, saw ${children.length}`);
    return children[index];
  };
  spawner.children = children;
  return spawner;
}

// Resolves once the child has received `count` stdin messages.
async function waitForWrites(child, count) {
  for (let i = 0; i < 200 && child.written.length < count; i++) await new Promise((r) => setTimeout(r, 5));
  assert.ok(child.written.length >= count, `expected ${count} writes, saw ${child.written.length}`);
}

function ask(spawnImpl, overrides = {}) {
  const tokens = [];
  const promise = streamClaudeCode({
    model: 'haiku',
    system: 'You are cue.',
    turns: [{ role: 'user', text: 'hello' }],
    maxTokens: 700,
    onToken: (t) => tokens.push(t),
    spawnImpl,
    ...overrides
  });
  return { promise, tokens };
}

test.afterEach(() => shutdown());

test('sends cue\'s turn as stream-json and streams the text deltas back', async () => {
  const spawner = fakeSpawner();
  const { promise, tokens } = ask(spawner);
  const child = await spawner.nth(0);
  await waitForWrites(child, 1);

  assert.deepEqual(child.written[0], { type: 'user', message: { role: 'user', content: 'hello' } });
  child.respond('all good');
  assert.equal(await promise, 'all good');
  assert.deepEqual(tokens, ['all ', 'good']);
});

test('passes the model, cue\'s system prompt and full config isolation on argv', async () => {
  const spawner = fakeSpawner();
  const { promise } = ask(spawner, { model: 'sonnet' });
  const child = await spawner.nth(0);
  const { args } = child.spawnArgs;

  assert.equal(args[args.indexOf('--model') + 1], 'sonnet');
  assert.match(args[args.indexOf('--system-prompt') + 1], /^You are cue\./);
  // The user's own CLAUDE.md, skills, hooks and MCP servers must never load.
  assert.equal(args[args.indexOf('--setting-sources') + 1], '');
  assert.ok(args.includes('--strict-mcp-config'));
  assert.ok(args.includes('--exclude-dynamic-system-prompt-sections'));
  // A screenshot assistant has no business touching the filesystem or network.
  const disallowed = args[args.indexOf('--disallowed-tools') + 1];
  for (const tool of ['Bash', 'Edit', 'Write', 'Read', 'WebFetch', 'Task']) assert.match(disallowed, new RegExp(tool));
  // An empty scratch cwd, so a repo cue floats over cannot leak its CLAUDE.md.
  assert.match(child.spawnArgs.options.cwd, /cue-claude-code/);

  child.respond('ok');
  await promise;
});

test('states the token budget in the prompt, since the CLI has no max_tokens flag', async () => {
  const spawner = fakeSpawner();
  const { promise } = ask(spawner, { maxTokens: 1400 });
  const child = await spawner.nth(0);
  const prompt = child.spawnArgs.args[child.spawnArgs.args.indexOf('--system-prompt') + 1];
  assert.match(prompt, /under roughly 980 words/);
  child.respond('ok');
  await promise;
});

test('attaches a screenshot to the final turn as a base64 image block', async () => {
  const spawner = fakeSpawner();
  const { promise } = ask(spawner, { image: { mime: 'image/png', b64: 'AAAB' } });
  const child = await spawner.nth(0);
  await waitForWrites(child, 1);

  assert.deepEqual(child.written[0].message.content, [
    { type: 'image', source: { type: 'base64', media_type: 'image/png', data: 'AAAB' } },
    { type: 'text', text: 'hello' }
  ]);
  child.respond('a screenshot');
  await promise;
});

test('falls back to the result text when no partial deltas arrive', async () => {
  const spawner = fakeSpawner();
  const { promise, tokens } = ask(spawner);
  const child = await spawner.nth(0);
  await waitForWrites(child, 1);
  child.respond('only the result', { partial: false });
  assert.equal(await promise, 'only the result');
  assert.deepEqual(tokens, ['only the result']);
});

test('keeps a warm standby so the next request does not pay CLI boot', async () => {
  const spawner = fakeSpawner();
  const first = ask(spawner);
  const child = await spawner.nth(0);
  await waitForWrites(child, 1);
  child.respond('one');
  await first.promise;

  // The used process is retired (its session now holds history) and a spare is
  // spawned off the critical path.
  assert.deepEqual(child.killSignals, ['SIGTERM']);
  const standby = await spawner.nth(1);
  assert.equal(standby.written.length, 0, 'a standby must be unused');

  // The next request claims that spare instead of spawning a third process.
  const second = ask(spawner);
  await waitForWrites(standby, 1);
  assert.equal(spawner.children.length, 2, 'the standby should have been reused');
  standby.respond('two');
  assert.equal(await second.promise, 'two');
});

test('never reuses a session across requests, so history cannot bleed', async () => {
  const spawner = fakeSpawner();
  const first = ask(spawner);
  const a = await spawner.nth(0);
  await waitForWrites(a, 1);
  a.respond('one');
  await first.promise;

  const second = ask(spawner, { turns: [{ role: 'user', text: 'second question' }] });
  const b = await spawner.nth(1);
  await waitForWrites(b, 1);
  // cue renders context into one self-contained turn per request, so the
  // second request must land in a process that has seen nothing else.
  assert.notEqual(a, b);
  assert.deepEqual(b.written, [{ type: 'user', message: { role: 'user', content: 'second question' } }]);
  b.respond('two');
  await second.promise;
});

test('gives a concurrent request its own process rather than queueing it', async () => {
  const spawner = fakeSpawner();
  const first = ask(spawner, { turns: [{ role: 'user', text: 'left' }] });
  const a = await spawner.nth(0);
  await waitForWrites(a, 1);
  const second = ask(spawner, { turns: [{ role: 'user', text: 'right' }] });
  const b = await spawner.nth(1);
  await waitForWrites(b, 1);

  assert.notEqual(a, b);
  b.respond('RIGHT');
  a.respond('LEFT');
  assert.deepEqual(await Promise.all([first.promise, second.promise]), ['LEFT', 'RIGHT']);
});

test('a different system prompt gets its own standby instead of evicting the first', async () => {
  const spawner = fakeSpawner();
  const first = ask(spawner, { system: 'Feature A' });
  const a = await spawner.nth(0);
  await waitForWrites(a, 1);
  a.respond('one');
  await first.promise;
  await spawner.nth(1); // standby for Feature A

  const second = ask(spawner, { system: 'Feature B' });
  const b = await spawner.nth(2); // A's standby is not usable for B
  await waitForWrites(b, 1);
  assert.match(b.spawnArgs.args[b.spawnArgs.args.indexOf('--system-prompt') + 1], /^Feature B/);
  b.respond('two');
  await second.promise;
});

test('reports a CLI that is missing rather than a raw spawn error', async () => {
  const spawner = fakeSpawner();
  const { promise } = ask(spawner, { cliPath: '/nope/claude' });
  const child = await spawner.nth(0);
  child.emit('error', Object.assign(new Error('spawn ENOENT'), { code: 'ENOENT' }));
  await assert.rejects(promise, /Claude Code CLI not found at "\/nope\/claude"/);
});

test('tells the user to finish /login when the session is not authenticated', async () => {
  const spawner = fakeSpawner();
  const { promise } = ask(spawner);
  const child = await spawner.nth(0);
  child.stderr.write('Invalid API key · Please run /login\n');
  await new Promise((r) => setTimeout(r, 10));
  child.emit('close', 1);
  await assert.rejects(promise, /not logged in/);
});

test('surfaces a usage-limit result as a switch-providers hint', async () => {
  const spawner = fakeSpawner();
  const { promise } = ask(spawner);
  const child = await spawner.nth(0);
  await waitForWrites(child, 1);
  child.fail('Claude AI usage limit reached');
  await assert.rejects(promise, /usage limit/);
});

test('rejects instead of hanging when the CLI dies mid-answer', async () => {
  const spawner = fakeSpawner();
  const { promise } = ask(spawner);
  const child = await spawner.nth(0);
  await waitForWrites(child, 1);
  child.emit('close', 1);
  await assert.rejects(promise, /Claude Code CLI exited with code 1/);
});

test('aborting a request kills its process', async () => {
  const spawner = fakeSpawner();
  const controller = new AbortController();
  const { promise } = ask(spawner, { signal: controller.signal });
  const child = await spawner.nth(0);
  await waitForWrites(child, 1);
  controller.abort();
  await assert.rejects(promise);
  assert.deepEqual(child.killSignals, ['SIGTERM']);
});

test('shutdown leaves no standby process running', async () => {
  const spawner = fakeSpawner();
  const { promise } = ask(spawner);
  const child = await spawner.nth(0);
  await waitForWrites(child, 1);
  child.respond('ok');
  await promise;
  const standby = await spawner.nth(1);

  shutdown();
  assert.deepEqual(standby.killSignals, ['SIGTERM']);
});

test('defaults track the newest Sonnet and Haiku through CLI aliases', () => {
  // Verified against the CLI: `sonnet` -> claude-sonnet-5, `haiku` ->
  // claude-haiku-4-5-20251001, while claude-sonnet-5-5 / claude-haiku-5-5 /
  // claude-haiku-5 are all rejected. Aliases mean a newer release is picked up
  // with no code change, so these must never be pinned ids.
  assert.equal(DEFAULT_MODELS.fast, 'haiku');
  assert.equal(DEFAULT_MODELS.smart, 'sonnet');
  for (const id of Object.values(DEFAULT_MODELS)) assert.doesNotMatch(id, /\d/);
});

test('an explicit CLI path wins over probing the install locations', () => {
  assert.equal(resolveCli('  /custom/claude  '), '/custom/claude');
  assert.ok(resolveCli('').length > 0);
});
