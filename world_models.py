import torch.nn as nn

class LunarLanderWorldModel(nn.Module):
    def __init__(self):
        super(LunarLanderWorldModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(8 + 4, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 8 + 1)
        )

    def forward(self, x):
        return self.network(x)

class CartpoleWorldModel(nn.Module):
    def __init__(self):
        super(CartpoleWorldModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(4 + 2, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 4 + 1)
        )

    def forward(self, x):
        return self.network(x)