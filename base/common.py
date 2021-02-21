import logging
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
