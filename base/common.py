import logging
import multiprocessing as mp
import timeit
from functools import wraps

def timer(function):
    @wraps(function)
    def new_function(*args, **kwargs):
        start_time = timeit.default_timer()
        try:
            return function(*args, **kwargs)
        finally:
            elapsed = timeit.default_timer() - start_time
            logging.info('Function "{name}" took {time} seconds to complete.'.format(name=function.__name__, time=elapsed))
    return new_function

def runInParallel(inputs, fnc, serial=False):
    if serial:
        return [fnc(*input) for input in inputs]
    with mp.Pool(processes=mp.cpu_count()) as pool:
        async_results = [pool.apply_async(fnc, (*input,)) for input in inputs]
        results = [async_result.get() for async_result in async_results]
    return results
