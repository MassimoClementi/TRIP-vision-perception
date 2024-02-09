# Date:     2024-01-27
# Author:   Massimo Clementi <massimo_clementi@icloud.com>
# Topic:    Utility functions and classes

from functools import wraps
import numpy as np
import time

def print_execution_time(func):
    @wraps(func)
    def print_execution_time_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'-> {func.__name__} took {(total_time*1e3):.0f} ms')
        return result
    return print_execution_time_wrapper

# Get index of the n-th occurence of True in the provided mask 
def GetNthOccurenceIndex(mask : np.array, nth : int) -> int:
    first_occurence = np.where(mask)
    if(len(first_occurence[0]) >= nth):
        return first_occurence[0][nth]
    else:
        return -1

