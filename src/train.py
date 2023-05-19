from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

from src.model import NLSModel
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR


class Train:
    def __init__(
        self,
        dataset: Dataset,
        ps: float,
        itg_window: float,
        itg_samples: int,
        lr: float,
        epochs: int,
        batch_size: int,
        steplr_step_size: int,
        steplr_gamma: float,
        loss_plot_path: str,
    ):
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.criterion = nn.MSELoss()
        self.model = NLSModel(
            ps=ps,
            itg_window=itg_window,
            itg_samples=itg_samples,
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = StepLR(
            self.optimizer, step_size=steplr_step_size, gamma=steplr_gamma
        )

        self.epochs = epochs
        self.loss_plot_path = loss_plot_path

    def run(self):
        losses = []
        best_model = None
        best_loss = float("inf")
        for epoch in range(self.epochs):
            self.model.train()  # Set the model to training mode
            running_loss = 0.0

            for inputs, targets in self.dataloader:
                self.optimizer.zero_grad()  # Zero the gradients

                outputs = self.model(*inputs)
                loss = self.criterion(outputs, targets)

                loss.backward()  # Perform backward pass
                self.optimizer.step()  # Update the model parameters

                running_loss += loss.item()

            self.scheduler.step()

            # Calculate and print the average loss for the epoch
            epoch_loss = running_loss / len(self.dataloader)
            losses.append(epoch_loss)
            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {epoch_loss:.4f}")

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model = self.model.state_dict()

        # draw loss plot
        plt.plot(range(self.epochs), losses)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.savefig(self.loss_plot_path)

        # load best model
        self.model.load_state_dict(best_model)
