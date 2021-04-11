import pandas as pd
from os.path import join, dirname, abspath


class Wili2018Dataset(object):
    '''
        Wili2018 dataset https://arxiv.org/pdf/1801.07779.pdf
    '''

    BASE_PATH = join(dirname(dirname(dirname(abspath(__file__)))), 'data', 'wili-2018')

    def __init__(self, language_subset=None):
        self.X_train = open(join(self.BASE_PATH, 'x_train.txt')).read().splitlines()
        self.Y_train = open(join(self.BASE_PATH, 'y_train.txt')).read().splitlines()

        self.X_test = open(join(self.BASE_PATH, 'x_test.txt')).read().splitlines()
        self.Y_test = open(join(self.BASE_PATH, 'y_test.txt')).read().splitlines()

        self.labels = pd.read_csv(join(self.BASE_PATH, 'labels.csv'), sep=';')

        if language_subset:
            self.__filter_to_language_subset(language_subset)

    def __filter_to_language_subset(self, language_subset):
        train_subset_idx = [idx for idx, lang in enumerate(self.Y_train) if lang in language_subset]
        test_subset_idx = [idx for idx, lang in enumerate(self.Y_test) if lang in language_subset]

        self.X_train = [self.X_train[i] for i in train_subset_idx]
        self.Y_train = [self.Y_train[i] for i in train_subset_idx]

        self.X_test = [self.X_test[i] for i in test_subset_idx]
        self.Y_test = [self.Y_test[i] for i in test_subset_idx]

    def get_data(self):
        return self.X_train, self.Y_train, self.X_test, self.Y_test

    def get_metadata(self):
        return self.labels
