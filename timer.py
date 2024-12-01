# Import the time library
import time

class Timer:
    def __init__(self, duration_in_nanos: int):
        self.duration_in_nanos = duration_in_nanos
        self.last_call = time.time_ns()
        self.start = None
        self.pause_time = None

    def remaining_time_nanos(self):
        return self.duration_in_nanos - (time.time_ns() - self.start)
    
    def elapsed_time_nanos(self):
        return time.time_ns() - self.last_call
    
    def pause(self):
        self.pause_time = time.time_ns()
    
    def resume(self):
        self.last_call = time.time_ns()
        if (self.start is None):
            self.start = self.last_call
        if (self.pause_time is not None):
            self.start += (self.last_call - self.pause_time)
            self.pause_time = None
        self.pause_time = None

    def did_buzz(self):
        return self.remaining_time_nanos() <= 0
    
    def pretty_print_time_remaining(self, timer):
        remaining, millis = divmod(timer.remaining_time_nanos() // 1000000, 1000) 
        mins, secs = divmod(remaining, 60)
        print("{:02d}:{:02d}:{:03d}".format(mins, secs, millis))