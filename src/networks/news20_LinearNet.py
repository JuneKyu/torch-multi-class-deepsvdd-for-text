import torch
import torch.nn as nn
import torch.nn.functional as F

from base.base_net import BaseNet


class News20_LinearNet(BaseNet):

    def __init__(self):
        super().__init__()

        # for sentence bert
        #  TODO: set rep_dim dynamically
        self.rep_dim = 128
        #  self.pool = nn.MaxPool1d()

        self.linear1 = nn.Linear(768, 500)
        self.linear2 = nn.Linear(500, 500)
        self.linear3 = nn.Linear(500, 500)
        self.linear4 = nn.Linear(500, 2000)
        self.linear5 = nn.Linear(2000, self.rep_dim)


    def forward(self, x):
        x = self.linear1(F.leaky_relu(x))
        x = self.linear2(F.leaky_relu(x))
        x = self.linear3(F.leaky_relu(x))
        x = self.linear4(F.leaky_relu(x))
        x = self.linear5(x)
        return x


class News20_LinearNet_Autoencoder(BaseNet):

    def __init__(self):
        super().__init__()

        self.rep_dim = 128
        
        # Encoder (must match the Deep SVDD network above)
        self.linear1 = nn.Linear(768, 500)
        nn.init.xavier_uniform_(self.linear1.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.linear2 = nn.Linear(500, 500)
        nn.init.xavier_uniform_(self.linear2.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.linear3 = nn.Linear(500, 500)
        nn.init.xavier_uniform_(self.linear3.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.linear4 = nn.Linear(500, 2000)
        nn.init.xavier_uniform_(self.linear4.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.linear5 = nn.Linear(2000, self.rep_dim)
        nn.init.xavier_uniform_(self.linear5.weight, gain=nn.init.calculate_gain('leaky_relu'))

        # Decoder
        self.linear6 = nn.Linear(self.rep_dim, 2000)
        nn.init.xavier_uniform_(self.linear6.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.linear7 = nn.Linear(2000, 500)
        nn.init.xavier_uniform_(self.linear7.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.linear8 = nn.Linear(500, 500)
        nn.init.xavier_uniform_(self.linear8.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.linear9 = nn.Linear(500, 500)
        nn.init.xavier_uniform_(self.linear9.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.linear10 = nn.Linear(500, 768)
        nn.init.xavier_uniform_(self.linear10.weight, gain=nn.init.calculate_gain('leaky_relu'))

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(F.leaky_relu(x))
        x = self.linear3(F.leaky_relu(x))
        x = self.linear4(F.leaky_relu(x))
        x = self.linear5(x)
        x = self.linear6(F.leaky_relu(x))
        x = self.linear7(F.leaky_relu(x))
        x = self.linear8(F.leaky_relu(x))
        x = self.linear9(F.leaky_relu(x))
        x = self.linear10(x)
        x = torch.sigmoid(x)
        return x
