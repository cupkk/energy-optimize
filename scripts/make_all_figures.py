from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    subprocess.run(
        [sys.executable, str(ROOT / "experiments" / "make_paper_figures.py")],
        cwd=ROOT,
        check=True,
    )


if __name__ == "__main__":
    main()
