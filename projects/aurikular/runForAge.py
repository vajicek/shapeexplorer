#!/usr/bin/python3

""" Run forAge and extract results. """

import json
import logging
import os
import subprocess
import multiprocessing as mp


from base.common import timer

PATH_TO_FORAGE = '../forAge/target/forage-1.0-SNAPSHOT.jar'
PROCESSES_PER_CPU = 1

@timer
def runForAge(input, no):
    logging.debug("pid=%s, no=%s", os.getpid(), no)
    cmd = ['java', '-jar', PATH_TO_FORAGE, '-i', input]
    result = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    return json.loads(result.stdout.read())

@timer
def runForAgeOnFiles(inputs):
    with mp.Pool(processes=mp.cpu_count() * PROCESSES_PER_CPU) as pool:
        async_results = [pool.apply_async(runForAge, (input, i)) for input, i in zip(inputs, range(len(inputs)))]
        results = [async_result.get() for async_result in async_results]
    return results

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    runForAge()
