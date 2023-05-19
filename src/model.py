import torch
from torch import nn


def nls_prob_dist(logt0, A, w, logt1, n, logt):
    F = (A / torch.pi) * (w / ((logt0 - logt1) ** 2 + w**2))
    return (1 - torch.e ** (-((10**logt / (10**logt0)) ** n))) * F


def nls(logt, A, w, logt1, n, l_bd, u_bd, itg_samples):
    logt0 = torch.linspace(l_bd, u_bd, itg_samples)

    pd = nls_prob_dist(logt0, A, w, logt1, n, logt)

    itg = torch.trapz(pd, logt0)

    return itg


class BridgeNet(nn.Module):
    def __init__(self):
        super(BridgeNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 10),
            nn.Sigmoid(),
            nn.Linear(10, 10),
            nn.Sigmoid(),
            nn.Linear(10, 1),
        )

    def forward(self, x):
        return self.net(x)


class NLSModel(nn.Module):
    def __init__(self, ps, itg_window, itg_samples):
        super(NLSModel, self).__init__()
        self.ps = ps
        self.itg_window = itg_window
        self.itg_samples = itg_samples

        self.n_net = nn.Sequential(
            BridgeNet(),
            nn.Softplus(),
        )
        self.w_net = nn.Sequential(
            BridgeNet(),
            nn.Softplus(),
        )
        self.logt1_net = BridgeNet()
        self.A_net = nn.Sequential(
            BridgeNet(),
            nn.Sigmoid(),
        )

    def forward(
        self,
        t: torch.Tensor,
        v: torch.Tensor,
    ):
        # Compute bridge parameters
        A = self.A_net(v)
        w = self.w_net(v)
        logt1 = self.logt1_net(v)
        n = self.n_net(v)

        # set up integration points
        lower_bound = torch.mean(logt1).item() - self.itg_window
        upper_bound = torch.mean(logt1).item() + self.itg_window
        itg_samples = self.itg_samples

        return (
            2
            * self.ps
            * nls(
                torch.log10(t),
                A,
                w,
                logt1,
                n,
                lower_bound,
                upper_bound,
                itg_samples,
            )
        )

    def bridge_params(self, v: torch.Tensor):
        import pandas as pd

        with torch.no_grad():
            A = self.A_net(v).reshape((-1,)).tolist()
            w = self.w_net(v).reshape((-1,)).tolist()
            logt1 = self.logt1_net(v).reshape((-1,)).tolist()
            n = self.n_net(v).reshape((-1,)).tolist()
            v = v.reshape((-1,)).tolist()

        rows = zip(v, A, w, logt1, n)
        cols = ["v", "A", "w", "logt1", "n"]
        df = pd.DataFrame(rows, columns=cols)

        return df
