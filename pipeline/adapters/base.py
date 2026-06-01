"""pipeline/adapters/base.py  -  Abstract translator interface."""
from abc import ABC, abstractmethod


class AbstractTranslator(ABC):
    """Abstract base class for all translators."""
    
    @property
    @abstractmethod
    def memory_usage_gb(self) -> float:
        """Estimated memory usage in GB (peak)."""
        pass
    
    @abstractmethod
    def translate(self, text: str) -> str:
        """Translate a text segment."""
        pass


def create_adapter_from_path(model_name: str, model_path=None):
    """Create adapter from model path (for local models)."""
    print(f"Adapter factory: would use {model_name} from {model_path or 'remote'}")
    return None


__all__ = ['AbstractTranslator', 'create_adapter_from_path']
