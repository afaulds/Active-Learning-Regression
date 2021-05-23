from model.ss_base import SemiSupervisedBase
from utils import Timer


class SemiSupervisedRandom(SemiSupervisedBase):

    def update_labeled(self):
        Timer.start("Random")
        self.labeled_pos_list.extend(self.unlabeled_pos_list[:self.batch_count])
        self.unlabeled_pos_list = self.unlabeled_pos_list[self.batch_count:]
        total_time = Timer.stop("Random")
