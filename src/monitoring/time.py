"""
time.py

Simple wall-clock timing utilities.
"""

import time
from contextlib import contextmanager 

@contextmanager
def timer():
    """
    Context manager for timing code blocks.
    """
    start = time.time()
    yield lambda: time.time() - start