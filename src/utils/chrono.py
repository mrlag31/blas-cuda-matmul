from time import time

class Chrono:
    start: float
    end: float
    
    def __enter__(self):
        self.start = time()
    
    def __exit__(self, _exc_type, _exc_value, _traceback):
        self.end = time()
    
    def print(self, file: str):
        t = str((self.end - self.start) * 1000).ljust(7, '0')[:7]
        print(f"[{file:<20}]: {t}ms")