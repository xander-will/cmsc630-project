# adapted from https://realpython.com/primer-on-python-decorators/#a-few-real-world-examples

import functools
import time

runtime_dict = {}

def timer(func):
    """Print the runtime of the decorated function"""
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()    # 1
        value = func(*args, **kwargs)
        end_time = time.perf_counter()      # 2
        run_time = end_time - start_time    # 3
        
        try:
            runtime_dict[func.__name__] += run_time
        except KeyError:
            runtime_dict[func.__name__] = run_time

        return value
    return wrapper_timer
