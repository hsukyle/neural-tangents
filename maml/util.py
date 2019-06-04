import numpy as onp

class Log(dict):
    def __init__(self, keys):
        for key in keys:
            self[key] = onp.array([])
    
    def append(self, keys_and_values):
        for (key, value) in keys_and_values:
            self[key] = onp.append(self[key], onp.array(value))