# WARNING! THIS FILE SHOULD NOT BE MODIFIED IN THIS REPO

from time import time
from datetime import datetime


def timeit(message="", add_timestamp=True, add_newline=False):
    def decorator(func):
        def wrapper(*args, **kwargs):
            point_time = time()
            print(f"[{datetime.now()}] " * add_timestamp + f"{message}...")
            out = func(*args, **kwargs)
            print(f"[{datetime.now()}] " * add_timestamp + f"{message}...done in {time() - point_time:.2f} sec.")
            if add_newline:
                print()
            return out

        return wrapper

    return decorator


if __name__ == "__main__":
    @timeit("Running run", False)
    def run(a):
        print(a)
        return 3


    a = run(4)
    print(a)
