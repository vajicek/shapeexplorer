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
def runForAge(input_file, i):
    logging.debug("pid=%s, i=%s, input=%s", os.getpid(), i, input_file)
    cmd = ['java', '-jar', PATH_TO_FORAGE, '-i', input_file]
    with subprocess.Popen(cmd, stdout=subprocess.PIPE) as result:
        return json.loads(result.stdout.read())


@timer
def runForAgeOnFiles(inputs):
    with mp.Pool(processes=mp.cpu_count() * PROCESSES_PER_CPU) as pool:
        async_results = [pool.apply_async(runForAge, (input_file, i))
                         for input_file, i in zip(inputs, range(len(inputs)))]
        results = [async_result.get() for async_result in async_results]
    return results


@timer
def runForAgeOnSample(input_sample):
    sample = input_sample.copy()
    results = runForAgeOnFiles([specimen['filename']
                                for specimen in sample['specimens']])
    for result, specimen in zip(results, sample['specimens']):
        specimen.update(result)
    return sample
