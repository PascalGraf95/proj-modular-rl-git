#!/usr/bin/env python
import ray
import time
import tensorflow as tf
import numpy as np
from multiprocessing import Process, Manager, Value, Array, Queue
import threading


class Buffer:
    def __init__(self):
        self.buffer = None

    def append_to_buffer(self, value):
        self.buffer.append(value)

    def read_buffer(self):
        return self.buffer[-5:]


class Actor:
    def __init__(self, buffer, device: str = '/cpu:0'):
        self.device = device
        self.counter = 0
        self.buffer = buffer


    def play_episodes_forever(self, buffer):
        while True:
            # Do something simulation
            time.sleep(0.5)
            # Write to buffer simulation
            buffer.append_to_buffer(self.counter)
            print("Appended", self.counter, "to Buffer.")
            self.counter += 1
        return 0


class Learner:
    def __init__(self, buffer):
        self.counter = 0
        self.buffer = buffer

    def learn_forever(self, buffer):
        while True:
            # Do something simulation
            time.sleep(5)
            # Read from buffer simulation
            values = buffer.read_buffer()
            print("Read from Buffer: ", values)
        return 0



if __name__ == '__main__':
    # ray.init(local_mode=True)
    buffer = Buffer()
    learner = Learner(buffer)
    actor = Actor(buffer)
    with Manager() as manager:
        # create the shared list
        buffer.buffer = manager.list()
        # instantiating process with arguments
        procs = [Process(target=actor.play_episodes_forever, args=(buffer, )), Process(target=learner.learn_forever, args=(buffer, ))]
        for proc in procs:
            proc.start()
        # complete the processes
        for proc in procs:
            proc.join()

    """
    threads = [threading.Thread(target=actor.play_episodes_forever), threading.Thread(target=learner.learn_forever)]
    # run all threads
    for thread in threads:
        thread.start()
    # wait for all threads to finish
    for thread in threads:
        thread.join()
    """
    # refs = []
    # refs.append(actor.play_episodes_forever.remote(buffer))
    # refs.append(learner.learn_forever.remote(buffer))
    # ray.wait(refs)
    print("Done")

