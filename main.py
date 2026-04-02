import sys
from pathlib import Path

# When run as a script (python /abs/path/main.py serve) the project root is
# not automatically on sys.path. Insert it so `from libucks.xxx import yyy`
# resolves correctly regardless of cwd or how Claude Desktop invokes us.
_PROJECT_ROOT = Path(__file__).parent.resolve()
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from libucks._cli import cli  # noqa: E402 — path bootstrap must come first

if __name__ == "__main__":
    cli()
