from time import time


class Timer:
    def __init__(self, quite: bool = False, *, desc: str):
        self.desc = desc
        self.quite = quite

    def __enter__(self):
        self.start_time = time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.quite:
            print(f'Time used to {self.desc}: {time() - self.start_time:.2f}s.')