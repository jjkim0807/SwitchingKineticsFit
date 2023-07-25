import math
from pathlib import Path
import torch
from torch.utils.data import Dataset
import pandas as pd


class SwitchingKineticsDataset(Dataset):
    def __init__(self, data_path: str) -> None:
        # find ps
        ps = None
        for file in Path(data_path).iterdir():
            if "pund" in file.name:
                file_data = pd.read_excel(file)
                ps = file_data["Qsw"].values[0]
                break
        assert ps is not None

        self.times = []
        self.voltages = []
        self.labels = []
        for file in Path(data_path).iterdir():
            if "hysteresis" in file.name or "pund" in file.name:
                continue

            voltage = file.name.split(" ")[4]
            voltage = float(voltage[:-1].replace("_", "."))
            self.voltages.append(torch.tensor([voltage]))

            time = file.name.split(" ")[5].split("ns")[0]
            time = float(time) * 1e-9
            self.times.append(torch.tensor([time]))

            file_data = pd.read_excel(file)
            label = float(file_data["RESULT"].values[0])
            label = label * (3.8e-9 / ps)
            self.labels.append(torch.tensor(label))

    def __len__(self):
        return len(self.times)

    def __getitem__(self, idx):
        return (self.times[idx], self.voltages[idx]), self.labels[idx]
