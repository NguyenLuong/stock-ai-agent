"""Import bridge for python-telegram-bot library.

The local 'telegram' package shadows the installed python-telegram-bot.
This module resolves the conflict by temporarily adjusting sys.path and
sys.modules to import the real library from site-packages.
"""

import os
import sys

_local_parent = os.path.realpath(os.path.join(os.path.dirname(__file__), os.pardir))

# Save local telegram modules currently in sys.modules
_saved_local = {}
for _key in list(sys.modules):
    if _key == "telegram" or _key.startswith("telegram."):
        _saved_local[_key] = sys.modules.pop(_key)

# Temporarily remove local package parent from sys.path
_orig_path = list(sys.path)
sys.path = [p for p in sys.path if os.path.abspath(p) != _local_parent]

try:
    from telegram import Bot  # noqa: F401
    from telegram.error import TelegramError  # noqa: F401
    from telegram.ext import Application  # noqa: F401
finally:
    sys.path = _orig_path
    # Restore saved local modules — they take priority for matching keys.
    # Real library modules whose keys are NOT in _saved_local stay in
    # sys.modules (e.g. telegram.error, telegram.ext) so the library
    # continues to work at runtime.
    sys.modules.update(_saved_local)
