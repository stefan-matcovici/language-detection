from torch.utils.data import Dataset


class Wili2018TorchDataset(Dataset):

    def __init__(self, X, y, word2idx, cls2idx, processor):
        self.X = X
        self.y = y

        self.word2idx = word2idx
        self.cls2idx = cls2idx
        self.processor = processor

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        tokenized_text = self.processor(self.X[index])
        return [self.word2idx.get(word, self.word2idx['<UNK>']) for word in tokenized_text], self.cls2idx[self.y[index]]
