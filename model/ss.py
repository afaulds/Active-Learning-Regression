from model.ss_bemcm import SemiSupervisedBEMCM
from model.ss_greedy import SemiSupervisedGreedy
from model.ss_random import SemiSupervisedRandom
from model.ss_qbc import SemiSupervisedQBC


class SemiSupervised:

    # Initialize specific selection option
    def __init__(self, type, X, y):
        if type == "random":
            self.ss = SemiSupervisedRandom(X, y)
        elif type == "qbc":
            self.ss = SemiSupervisedQBC(X, y)
        elif type == "greedy":
            self.ss = SemiSupervisedGreedy(X, y)
        elif type == "bemcm":
            self.ss = SemiSupervisedBEMCM(X, y)
        else:
            print("Incorrect type {}".format(type))
            exit()

    # Pass through to selected class.
    def is_done(self):
        return self.ss.is_done()

    def get_percent_labeled(self):
        return self.ss.get_percent_labeled()

    def get_labeled(self):
        return self.ss.get_labeled()

    def get_test(self):
        return self.ss.get_test()

    def update_labeled(self):
        self.ss.update_labeled()
