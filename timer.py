# Import the time library
import time

# Calculate the start time
start = time.time()

# Code here

# Calculate the end time and time taken
end = time.time()
length = end - start

class Timer:
    def __init__(self, duration_in_nanos: int):
        self.duration_in_nanos = duration_in_nanos
        self.last_call = time.time_ns()
        self.start = time.time_ns()

    def remaining_time_nanos(self):
        self.last_call = time.time_ns()
        return self.duration_in_nanos - (time.time_ns() - self.start)
    
    def elapsed_time_nanos(self):
        return time.time_ns() - self.start