import sys
from unittest.mock import MagicMock

for mod in [
    'sedna',
    'sedna.common',
    'sedna.common.class_factory',
    'sedna.common.config',
    'sedna.core',
    'sedna.algorithms'
]:
    if mod not in sys.modules:
        sys.modules[mod] = MagicMock()
