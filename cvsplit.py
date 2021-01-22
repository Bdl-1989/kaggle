from importconfig import *


class CVSplit():
    def __init__(self, folds):
        self.folds = folds

    def split(self):
        Fold = StratifiedKFold(n_splits=CFG.n_fold, shuffle=True, random_state=CFG.seed)
        for n, (train_index, val_index) in enumerate(Fold.split(self.folds, self.folds[CFG.target_col])):
            self.folds.loc[val_index, 'fold'] = int(n)
        self.folds['fold'] = self.folds['fold'].astype(int)
        print(folds.groupby(['fold', CFG.target_col]).size())