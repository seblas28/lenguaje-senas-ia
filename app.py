"""Local CLI entry point. The old Flask receiver and automatic publishing are retired."""
from kai_local.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
