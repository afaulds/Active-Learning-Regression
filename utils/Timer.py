import math
import time

class Timer:
    """
    Local variable that holds the start time by key.
    This code handles multiple starts as long as the
    key is unique.
    """
    start_time = {}

    @staticmethod
    def reset(key = "", message = ""):
        """
        Function to start timer using the provided key
        and output description of what
        is about to happen.  If no key is supplied
        we use a global timer.

        Args:
          key - Unique value for timer.
            message - Message to display.
        """
        Timer.start_time[key] = {
          "time": time.time(),
          "duration": 0,
          "count": 0,
        }
        if message != "":
            print(message)

    @staticmethod
    def start(key = "", message = ""):
        """
        Function to start timer using the provided key
        and output description of what
        is about to happen.  If no key is supplied
        we use a global timer.

        Args:
          key - Unique value for timer.
            message - Message to display.
        """
        if key not in Timer.start_time:
            Timer.reset(key)
        Timer.start_time[key]["time"] = time.time()
        if message != "":
            print(message)

    @staticmethod
    def stop(key = ""):
        """
        Function to end the timer and display how long the process
        took.  Prints message if it is included.

        Args:
          key - Unique value for timer.
        Return:
          None
        """
        if key in Timer.start_time:
            duration = time.time() - Timer.start_time[key]["time"]
        else:
            duration = 0
        Timer.start_time[key]["duration"] = Timer.start_time[key]["duration"] + duration
        Timer.start_time[key]["count"] += 1
        return duration

    @staticmethod
    def display(key = "", message = ""):
        """
        Function to end the timer and display how long the process
        took.  Prints message if it is included.

        Args:
          message - Message to display.
        Return:
          None
        """
        duration = Timer.start_time[key]["duration"]
        count = Timer.start_time[key]["count"]
        min = math.floor(duration / 60)
        sec = round(duration, 4) - min * 60
        duration_str = "%d:%05.2f" % (min, sec)
        print(key + " " + duration_str + " " + str(count) + " Calls")
