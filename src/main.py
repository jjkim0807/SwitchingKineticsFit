import torch
from src.dataset import SwitchingKineticsDataset
from src.eval import Eval
from src.train import Train
import fire


class Main:
    def __init__(self) -> None:
        pass

    def run(
        self,
        data_path: str = "data/normalized_001HZO.xlsx",
        graph_plot_path: str = "output/graph_plot.png",
        loss_plot_path: str = "output/loss_plot.png",
        ps: float = 0.5,
        itg_window=9,
        itg_samples=1000,
        epochs: int = 300,
        batch_size: int = 10,
        steplr_step_size: int = 50,
        steplr_gamma: float = 0.9,
        lr: float = 0.01,
    ):
        torch.set_default_dtype(torch.float64)

        ds = SwitchingKineticsDataset(data_path)

        train = Train(
            ds,
            ps=ps,
            lr=lr,
            itg_window=itg_window,
            itg_samples=itg_samples,
            epochs=epochs,
            batch_size=batch_size,
            steplr_step_size=steplr_step_size,
            steplr_gamma=steplr_gamma,
            loss_plot_path=loss_plot_path,
        )
        train.run()

        eval = Eval(
            ds,
            train.model,
            graph_plot_path=graph_plot_path,
        )
        eval.run()


if __name__ == "__main__":
    fire.Fire(Main)
