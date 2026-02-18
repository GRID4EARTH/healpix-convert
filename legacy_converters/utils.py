import warnings
from contextlib import contextmanager


@contextmanager
def suppress_warning(warning):
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(category=warning, action="ignore")
            yield
    finally:
        pass
