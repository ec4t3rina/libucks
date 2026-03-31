# libucks — Quickstart

## 1. Install

```bash
pip install -e ".[dev]"
```

## 2. Required Environment Variables

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

That is the only secret required. `TextStrategy.from_env()` reads it via the `anthropic` SDK default.

## 3. Init the current repo

```bash
libucks init --local "$(pwd)"
```

## 4. Serve the MCP bridge

```bash
libucks serve
```

Starts the MCP server on **stdio**. Register it in your MCP client (e.g. Claude Code `~/.claude/settings.json`):

```json
{
  "mcpServers": {
    "libucks": {
      "command": "libucks",
      "args": ["serve"]
    }
  }
}
```

## 5. Init a different local repo

```bash
libucks init --local /absolute/path/to/other/repo
```

Then run `libucks serve` from inside that repo (it reads `.libucks/` relative to cwd).

## 6. PYTHONPATH note

**Not required** if installed with `pip install -e .` — the `libucks` console script is on `PATH` and the package is importable.

If running the entry point directly (e.g. `python main.py serve`), set:

```bash
PYTHONPATH=. python main.py serve
```
