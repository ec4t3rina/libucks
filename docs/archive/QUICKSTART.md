# libucks — Quickstart (macOS, /Users/ecaterina/Developer/libucks)

## 1. Create & activate the virtual environment

```bash
cd /Users/ecaterina/Developer/libucks
python3 -m venv .venv
source .venv/bin/activate
```

## 2. Install the package (editable, with dev deps)

```bash
pip install -e ".[dev]"
```

This installs the `libucks` console script and makes `from libucks.xxx import yyy`
importable from anywhere on this Python interpreter.

## 3. Set your API key

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

`TextStrategy.from_env()` reads it via the `anthropic` SDK default. You only need
this in your shell for manual runs; for Claude Desktop see §5 below.

## 4. Index a repository

```bash
libucks init --local /Users/ecaterina/Developer/libucks
```

This walks the repo, embeds every source file, clusters the chunks, and writes
`.libucks/buckets/` and `.libucks/registry.json` into the target directory.

## 5. Claude Desktop config

Write the following to
`/Users/ecaterina/Library/Application Support/Claude/claude_desktop_config.json`
(create the file if it does not exist — Claude Desktop will not overwrite it on
launch if it is already present):

```json
{
  "mcpServers": {
    "libucks": {
      "command": "/Users/ecaterina/Developer/libucks/.venv/bin/python",
      "args": ["/Users/ecaterina/Developer/libucks/main.py", "serve"],
      "env": {
        "ANTHROPIC_API_KEY": "sk-ant-YOUR-KEY-HERE",
        "PYTHONPATH": "/Users/ecaterina/Developer/libucks"
      }
    }
  }
}
```

**Important notes:**
- Use absolute paths — Claude Desktop does not inherit your shell environment.
- `ANTHROPIC_API_KEY` **must** be in `"env"` for the same reason.
- `PYTHONPATH` is a belt-and-suspenders fallback; the venv Python is the primary
  mechanism.
- After saving, **quit and relaunch** Claude Desktop for the config to take effect.

## 6. Verify the server starts

From a fresh terminal (no venv active, any cwd):

```bash
/Users/ecaterina/Developer/libucks/.venv/bin/python \
  /Users/ecaterina/Developer/libucks/main.py serve
```

It should block silently, waiting for MCP stdin. Press `Ctrl-C` to stop.
No "Loading weights" or other text should appear on stdout — those messages are
redirected to stderr.

## 7. Index a different repo

```bash
libucks init --local /absolute/path/to/other/repo
```

Then update the `"args"` in `claude_desktop_config.json` to point `serve` at the
repo that contains `.libucks/` (the server reads `.libucks/` relative to cwd, or
from `paths.repo_root` in `.libucks/config.toml`).

---

## Phase 6: Production-Grade Dynamic Engine

### Step 1 — Start the server

```bash
libucks serve
```

This starts the MCP server (stdio) **and** automatically starts two background
tasks inside the same process:

| Background task | What it does | Frequency |
|---|---|---|
| `GitHookReceiver` | Listens on `.libucks/server.sock` for git events | Always-on |
| `HealthMonitor` | Splits overflowing buckets, merges redundant ones | Every 5 min |

### Step 2 — Install git hooks in a repo

Run once per repository you want libucks to track:

```bash
cd /path/to/your/repo
libucks install-hooks
```

This **appends** (never overwrites) three trigger lines to `.git/hooks/`:

```
post-commit     → libucks hook post-commit "$@" || true
post-checkout   → libucks hook post-checkout "$@" || true
post-rewrite    → libucks hook post-rewrite "$@" || true
```

After a `git commit` or `git checkout`, the hook sends a JSON event over the
Unix socket. The server replays any missed diffs and updates `last_indexed_head`.

### Step 3 — Verify the engine is running

Use the `libucks_status` MCP tool from Claude Desktop, or call it directly:

```bash
echo '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"libucks_status","arguments":{}}}' \
  | libucks serve 2>/dev/null
```

The JSON response includes `bucket_count` and `total_tokens`. A healthy system
shows the bucket count stabilising over time as the HealthMonitor splits and
merges.

**Background watcher confirmation** — check the server stderr log for:

```
[libucks] git_hook_receiver.listening  sock=.libucks/server.sock
[libucks] health_monitor.started       interval=300
```

If you see both lines, all three layers (JIT staleness, git hooks, health monitor)
are active.
