"""``python main.py`` — the same as ``python -m jarvis`` (any arguments too).

Kept so the old way of starting JARVIS still works; the assistant itself is the
``jarvis`` package.
"""

import sys

from jarvis.__main__ import main

if __name__ == "__main__":
    sys.exit(main())
