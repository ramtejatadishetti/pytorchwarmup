import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import torch.autograd.function


parser = argparse.ArgumentParser(description='MLP example')
parser.add_argument('--input-dim', type=int, default=1, metavar='I',
                        help='input dimension size')
parser.add_argument('--hidden-dim', type=int, default=1, metavar='H',
                        help='hidden dimension size')
parser.add_argument('--output-dim', type=int, default=1, metavar='O',
                        help='ouput dimension size')
args = parser.parse_args()

class NullyifyGrad(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        return input
    
    @staticmethod
    def backward(ctx, grad_output):
        return torch.zeros_like(grad_output)

class MlpNet(nn.Module):
    def __init__(self, input_dim=4,hidden_dim=4,output_dim=4):
        super(MlpNet, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=False)
#        self.fc2 = nn.Linear(hidden_dim, output_dim, bias=False)

        # We could change weight initialization
        # Ex: nn.init.<initialization function>(self.fc*.weight) refer https://pytorch.org/docs/master/nn.html#torch-nn-init

    def forward(self, x):
        x = self.fc1(x)
#        x = NullyifyGrad.apply(x)
        x = F.relu(x)
#        x = self.fc2(x)
#        x = NullyifyGrad.apply(x)
        return x


def get_output(inp):
    output = []
    for i in range(len(inp)):
        output.append(inp[i]*inp[i])

    return output




random_input = torch.rand(100,args.input_dim)
random_output = get_output(random_input)

criterion = nn.MSELoss()

net = MlpNet(input_dim=args.input_dim, hidden_dim=args.hidden_dim, output_dim=args.output_dim)
print(net)

params = list(net.parameters())
print(params)

optimizer = optim.SGD(net.parameters(), lr=0.0001)
for j in range(1000): 
    for i in range(len(random_input)):
        optimizer.zero_grad()

        pred = net(random_input[i])
        #print(pred.shape, random_output[i].shape)
        loss = criterion(pred, random_output[i])

        loss.backward()

        optimizer.step()

    #print(net.fc1.weight)
    #print(loss)

print(net.fc1.weight)
test_input = torch.rand(50,args.input_dim)


threshould = 0.01
counter = 0

#print(net)
test_output = get_output(test_input)

for i in range(len(test_input)):
    pred = net(test_input[i])
    #print(pred.data)
    print(pred.data, test_output[i].data, test_input[i].data)
    dif = (pred.data - test_output[i].data)
    if dif < 0:
        dif = dif * -1

    if dif < threshould:
        counter += 1

print(counter*100/len(test_input))



