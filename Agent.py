import torch
import torch.nn.functional as F
from torch import nn


class Agent:
    def __init__(self, model, lr):
        self.model = model
        self.opt = torch.optim.Adam(params=self.model.parameters(), lr=lr)

    def get_action(self, obs):
        obs = torch.from_numpy(obs).float().permute(0, 3, 1, 2)
        with torch.no_grad():
            logit = self.model(obs)
            categorical = torch.distributions.categorical.Categorical(logits=logit[0])
            act = categorical.sample()
            return act

    def update(self, d_x, d_y):
        d_x = torch.from_numpy(d_x).float().permute(0, 3, 1, 2)
        d_y = torch.from_numpy(d_y).long()

        if torch.cuda.is_available():
            d_x = d_x.cuda()
            d_y = d_y.cuda()

        logits = self.model(d_x)
        tmp = F.cross_entropy(logits, d_y)
        loss = torch.mean(tmp)

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return loss


class CNN(nn.Module):
    def __init__(self, n_count):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3 * n_count, 16, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 3))
        self.model[-1].weight.detach().zero_()
        self.model[-1].bias.detach().zero_()

    def forward(self, obs):
        return self.model(obs)
