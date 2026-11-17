import pickle
import os
import sys

def save(obj, name):
    print(f"Saving object {name}")
    print(f"Size of the object: {sys.getsizeof(obj) / 1000000:.2f} MB")
    with open(f"pantry/{name}.pickle", "wb") as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load(name):
    print(f"Loading object {name}")
    if os.path.isfile(f"pantry/{name}.pickle"):
        with open(f'pantry/{name}.pickle', 'rb') as file:
            return pickle.load(file)
    else:
        print("No such file")

    return None