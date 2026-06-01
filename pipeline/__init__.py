"""Pipeline package."""
# Minimal __init__ to allow imports - each module is independent
__version__ = "1.0.0"

# Optional re-exports if needed
from .chunker import *
from .register import *
from .alignment import *
from .translator import *
from .output import *
from .errors import *
