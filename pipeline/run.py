#!/usr/bin/env python3
"""pipeline/run.py  -  Wrapper to use epub.py as main entry point."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    # Use epub.py as the main translation entry point (handles model loading)
    from epub import main as epub_main
    
    return epub_main() or 0


if __name__ == "__main__":
    sys.exit(main() or 0)
