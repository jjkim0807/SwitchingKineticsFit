import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt


class Eval:
    def __init__(
        self,
        dataset: Dataset,
        model: nn.Module,
        graph_plot_path: str,
    ):
        self.model = model
        self.dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
        self.criterion = nn.MSELoss()
        self.graph_plot_path = graph_plot_path

    def run(self):
        plt.figure()
        ax = plt.axes(projection="3d")

        log_time = None
        voltage = None
        delta_p = None
        pred_delta_p = None

        self.model.eval()
        with torch.no_grad():
            test_loss = 0.0
            for inputs, targets in self.dataloader:
                outputs = self.model(*inputs)
                loss = self.criterion(outputs, targets)
                test_loss = loss.item()

                log_time = torch.log10(inputs[0]).reshape((-1,)).tolist()
                voltage = inputs[1].reshape((-1,)).tolist()
                delta_p = targets.tolist()
                pred_delta_p = outputs.tolist()

                # since voltages has discrete values,
                # find each values of voltages
                # and find the corresponding bridge parameters for each voltage

                discrete_voltages = torch.tensor(sorted(set(voltage))).reshape((-1, 1))
                bridge_params, E0 = self.model.bridge_params(discrete_voltages)

            print(f"Test Loss: {test_loss:.4f}")
            print(f"Bridge Parameters: \n")
            print(bridge_params)
            print(f"E0: {E0}")

        ax.scatter3D(log_time, voltage, delta_p, "gray", marker="o")
        ax.scatter3D(log_time, voltage, pred_delta_p, "red", marker="^")

        ax.set_xlabel("log(time)")
        ax.set_ylabel("voltage")
        ax.set_zlabel("delta P(t)")
        plt.savefig(self.graph_plot_path)
