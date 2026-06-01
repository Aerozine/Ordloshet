"""pipeline/errors.py  -  Enhanced error handling and reporting."""

import traceback
from typing import Optional


class TranslationError(Exception):
    """Base exception for translation errors with detailed context."""
    
    def __init__(self, message: str, chapter_name: str = None, 
                 text_sample: str = None, error_type: str = "generic"):
        self.message = message
        self.chapter_name = chapter_name
        self.text_sample = text_sample
        self.error_type = error_type
        
        # Get detailed traceback if available
        if error_type == "traceback":
            tb_lines = traceback.format_exc().split('\n')
            details = '\n'.join(tb_lines[-10:])  # Last 10 lines of traceback
        else:
            details = ""
        
        super().__init__(self.message)
    
    def __str__(self):
        return f"TranslationError[{self.error_type}]: {self.message}"


class TranslationTimeoutError(TranslationError):
    """Raised when translation takes too long."""
    error_type = "timeout"
    
    def __init__(self, message: str, chapter_name: str = None, 
                 elapsed_time: float = None, max_time: float = 300):
        super().__init__(message, chapter_name)
        self.elapsed_time = elapsed_time or 300
        self.max_time = max_time
        
    def __str__(self):
        return (f"TranslationTimeoutError[{self.error_type}]: "
                f"{self.message}\n"
                f"  Elapsed: {self.elapsed_time:.1f}s / Max: {self.max_time:.1f}s")


class GlossaryNotFoundError(TranslationError):
    """Raised when required glossary term is missing."""
    error_type = "glossary_missing"
    
    def __init__(self, term: str, context: str = None):
        super().__init__(f"Glossary term not found: '{term}'", 
                        error_type="glossary_missing")
        self.term = term
        self.context = context
    
    def __str__(self):
        result = f"TranslationError[glossary_missing]: Glossary term not found: '{self.term}'"
        if self.context:
            result += f"\n  Context: {self.context}"
        return result


class RegisterMismatchError(TranslationError):
    """Raised when register (tu/vous) detection failed."""
    error_type = "register_mismatch"
    
    def __init__(self, detected: str = None, expected: str = None, 
                 chapter_name: str = None):
        super().__init__(f"Register mismatch detected", chapter_name=chapter_name)
        self.detected = detected
        self.expected = expected
    
    def __str__(self):
        result = f"TranslationError[register_mismatch]: Register mismatch detected"
        
        if self.detected:
            result += f"\n  Detected: {self.detected}"
        
        if self.expected:
            result += f"\n  Expected: {self.expected}"
        
        return result


def report_error(error: TranslationError, log_file = None) -> str:
    """Generate human-readable error report.
    
    Args:
        error: The translation error to report
        log_file: Optional file path to append error to
    
    Returns:
        Formatted error message string
    """
    lines = [f"Error: {error}"]
    
    if hasattr(error, 'chapter_name') and error.chapter_name:
        lines.append(f"  Chapter: {error.chapter_name}")
    
    if hasattr(error, 'text_sample') and error.text_sample:
        # Truncate long samples
        sample = error.text_sample.strip()
        if len(sample) > 200:
            sample = sample[:197] + "..."
        lines.append(f"  Sample: {sample}")
    
    if hasattr(error, 'elapsed_time') and error.elapsed_time:
        lines.append(f"  Duration: {error.elapsed_time:.1f}s")
    
    if log_file:
        try:
            with open(log_file, 'a') as f:
                f.write('\n'.join(lines) + '\n\n')
        except (IOError, OSError):
            pass
    
    return '\n'.join(lines)


def safe_translate(translate_fn, text: str, timeout: float = 300, 
                   on_error=None) -> Optional[str]:
    """Safely translate text with error handling.
    
    Args:
        translate_fn: Translation function
        text: Text to translate
        timeout: Maximum seconds to wait (default 300)
        on_error: Optional callback for errors (str or None)
    
    Returns:
        Translated text or original if translation failed
    """
    import signal
    
    def timeout_handler(signum, frame):
        raise TimeoutError("Translation timed out")
    
    # Set up timeout
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(int(timeout))
    
    try:
        return translate_fn(text)
    except TimeoutError:
        if on_error:
            error_msg = on_error("timeout")
            if callable(error_msg):
                return error_msg()
            return error_msg
        raise
    except Exception as e:
        signal.alarm(0)  # Cancel timeout
        
        # Log error if we have a logger available
        try:
            from .errors import report_error, TranslationError
            import os
            from pathlib import Path
            
            output_path = Path("output") / "translate_errors.log"
            
            if hasattr(e, 'chapter_name'):
                chapter_name = e.chapter_name
            else:
                chapter_name = "unknown"
            
            error = TranslationError(
                f"{type(e).__name__}: {str(e)}",
                chapter_name=chapter_name,
                text_sample=text[:200] if len(text) > 200 else text
            )
            
            report_error(error, str(output_path))
        except:
            pass  # Don't fail on logging errors
        
        if on_error:
            error_msg = on_error(str(type(e).__name__))
            if callable(error_msg):
                return error_msg()
            return error_msg
        
        raise
