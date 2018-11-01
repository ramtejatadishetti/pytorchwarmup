import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse


parser = argparse.ArgumentParser(description='MLP example')
parser.add_argument('--input-dim', type=int, default=4, metavar='I',
                        help='input dimension size')
parser.add_argument('--hidden-dim', type=int, default=4, metavar='H',
                        help='hidden dimension size')
parser.add_argument('--output-dim', type=int, default=4, metavar='O',
                        help='ouput dimension size')
args = parser.parse_args()

class MlpNet(nn.Module):
    def __init__(self, input_dim=4,hidden_dim=4,output_dim=4):
        super(MlpNet, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, output_dim, bias=False)

        # We could change weight initialization
        # Ex: nn.init.<initialization function>(self.fc*.weight) refer https://pytorch.org/docs/master/nn.html#torch-nn-init

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


random_input = torch.randn(100,args.input_dim)
random_output = torch.randn(100,args.output_dim)
criterion = nn.MSELoss()

net = MlpNet(input_dim=args.input_dim, hidden_dim=args.hidden_dim, output_dim=args.output_dim)
#print(net)

params = list(net.parameters())
#print(params)

optimizer = optim.SGD(net.parameters(), lr=0.01)
for j in range(100): 
    for i in range(100):
        optimizer.zero_grad()

        pred = net(random_input[i])
        #print(pred.shape, random_output[i].shape)
        loss = criterion(pred, random_output[i])

        loss.backward()

        optimizer.step()

        #print(net.fc1.weight, net.fc2.weight)
        print(loss)
