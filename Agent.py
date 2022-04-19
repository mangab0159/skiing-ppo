import torch
import torchvision.transforms.functional as vision_F
from torch import nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical


class Agent(nn.Module):
    def __init__(self, lr):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2592, 256),  # 13824, 2592
            nn.ReLU(),
            nn.Linear(256, 3))
        self.model[-1].weight.detach().zero_()
        self.model[-1].bias.detach().zero_()

        self.opt = torch.optim.Adam(params=self.model.parameters(),
                                    lr=lr)
        self._eps = 1e-25

        # use_cuda = torch.cuda.is_available()
        # if use_cuda:
        #     self.model = self.model.cuda()

    def forward(self, obs):
        obs_resize = vision_F.resize(obs, (84, 84))
        return self.model(obs_resize)

    def get_action(self, x):
        with torch.no_grad():
            logit = self(x)
            act = torch.distributions.categorical.Categorical(logits=logit[0]).sample()  # .numpy()
            return act

    def update_episodes(self, states, actions, returns, use_norm=False):
        # episode batch update version of REINFORCE

        if use_norm:
            returns = (returns - returns.mean()) / (returns.std() + self._eps)
        logits = self(states)
        logp = F.log_softmax(logits, -1) # [batch_size, action_size]
        logp = (logp * F.one_hot(actions, 3)).sum(1) # [batch_size]
        # dist = Categorical(logits=logits)
        # prob = dist.probs[range(states.shape[0]), actions]

        self.opt.zero_grad()

        # compute policy gradient loss
        #pg_loss = - torch.log(prob + self._eps) * returns.squeeze()  # [num. steps x 1]
        pg_loss = -logp * returns
        pg_loss = pg_loss.mean()  # [1]
        pg_loss.backward()

        self.opt.step()
