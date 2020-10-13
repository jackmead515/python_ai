import pickle
import os

def load_pickle(file_name):
    if os.path.isfile(file_name):
        with open(file_name, 'rb') as f:
            return pickle.load(f)

def save_pickle(file_name, data):
    with open(file_name, 'wb') as f:
        pickle.dump(data, f)