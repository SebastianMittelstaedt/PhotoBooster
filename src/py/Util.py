import time


class Stopwatch:
    def __init__(self):
        self.t1 = 0.0
        self.t2 = 0.0

    def start(self):
        self.t1 = time.time()
        self.t2 = self.t1

    def check(self, text):
        t = time.time()
        dt = t - self.t2
        print("\t %s took: %s" % (text, str(dt)))
        self.t2 = time.time()

    def end(self):
        t = time.time()
        dt = t - self.t1
        print("\t The whole process took: %s" % (str(dt)))