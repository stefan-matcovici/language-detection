import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder


class Wili2018TransformerDataset(Dataset):
    def __init__(self, tokenizer, X, y):
        self.encodings = tokenizer(X, truncation=True, padding=True)

        self.label_encoder = LabelEncoder()
        self.labels = self.label_encoder.fit_transform(y)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
