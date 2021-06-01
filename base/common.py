"""Common code for all shape explorer projects."""
import logging
import multiprocessing as mp
import timeit
from functools import wraps, partial

def timer(function=None, level=logging.DEBUG):
    if function is None:
        return partial(timer, level=level)

    @wraps(function)
    def wrapper(*args, **kwargs):
        start_time = timeit.default_timer()
        try:
            return function(*args, **kwargs)
        finally:
            elapsed = timeit.default_timer() - start_time
            message = 'Function "%s" took %f seconds to complete.'
            logging.log(level, message, function.__name__, elapsed)
    return wrapper

def runInParallel(inputs, fnc, serial=False):
    if serial:
        return [fnc(**input) for input in inputs]
    with mp.Pool(processes=mp.cpu_count()) as pool:
        async_results = [pool.apply_async(fnc, (), input) for input in inputs]
        results = [async_result.get() for async_result in async_results]
    return results
