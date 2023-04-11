from torch.utils.data import Dataset, DataLoader


class MyDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass