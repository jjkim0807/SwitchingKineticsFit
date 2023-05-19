import torch
from torch.utils.data import Dataset
import pandas as pd


class SwitchingKineticsDataset(Dataset):
    def __init__(
        self,
        data_path: str,
    ) -> None:
        data_frame = pd.read_excel(data_path)

        voltages = [torch.tensor([float(x[:-1])]) for x in data_frame.columns[1:]]

        self.times = []
        self.voltages = []
        self.labels = []
        for datapoint in data_frame.values:
            time = torch.tensor([datapoint[0]])
            for i, p in enumerate(datapoint[1:]):
                self.times.append(time)
                self.voltages.append(voltages[i])
                self.labels.append(torch.tensor(p))

    def __len__(self):
        return len(self.times)

    def __getitem__(self, idx):
        return (self.times[idx], self.voltages[idx]), self.labels[idx]
