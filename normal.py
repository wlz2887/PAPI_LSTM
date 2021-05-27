import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import argparse
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('AGG')

x = np.load('./mpx_data.npy')
y = np.load('./ocoe_data.npy')
#train_x = torch.unsqueeze(torch.from_numpy(x[0]),dim=1)
#test_x = torch.unsqueeze(torch.from_numpy(x[1]),dim=1)
#test_y = torch.unsqueeze(torch.from_numpy(y[1]),dim=1)
#train_y = torch.unsqueeze(torch.from_numpy(y[0]),dim=1)

parser = argparse.ArgumentParser()
parser.add_argument('--method',type = int,default = 0) #0——默认方法 1——DNN结果
parser.add_argument('--event_id',type=int,default = 0) #选取事件
args = parser.parse_args()

class net(nn.Module):
    def __init__(self,in_dim,n_hidden_1,n_hidden_2,out_dim):
        super(net,self).__init__()
        self.layer1 = nn.Linear(in_dim,n_hidden_1)
        self.layer2 = nn.Linear(n_hidden_1,n_hidden_2)
        self.predict = nn.Linear(n_hidden_2,out_dim)

    def forward(self,x):
        out = nn.functional.relu(self.layer1(x))
        out = nn.functional.relu(self.layer2(out))
        out = self.predict(out)
        return out

net = net(in_dim=1,n_hidden_1=64,n_hidden_2=32,out_dim=1)
net = net.double()
optimizer = torch.optim.Adam(net.parameters(),lr=0.01)
loss_func = torch.nn.MSELoss()

def train(epoch):
    for t in range(epoch):
        prediction = net(train_x)
        loss = loss_func(prediction,train_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if(t%200==1):
            print(loss)
            #if(t==3999):
            #   torch.save(net,'./kmeans.pkl')

def test(turn):    
    test_x = torch.unsqueeze(torch.from_numpy(x[turn]),dim=1)
    test_y = torch.unsqueeze(torch.from_numpy(y[turn]),dim=1)
    x_default = torch.squeeze(test_x)
    x_default = x_default.detach().numpy()
    Loss=0
    Loss2=0
    A,B,C = [],[],[]
    new_y=torch.squeeze(test_y)
    new_y = new_y.detach().numpy()

    if(args.method == 1):
        pre = torch.squeeze(net(test_x))
        pre = pre.detach().numpy()
        np.save('./pred.npy',pre)
        for i,j in zip(pre,new_y):
            a = pow(10,i)
            b = pow(10,j)
            #A.append(a)
            #B.append(b)
            Loss += abs(a-b)/b
        return Loss/pre.size

    else:
        for i,j in zip(new_y,x_default):
            a = pow(10,i)
            b = pow(10,j)
            Loss2 += abs(a-b)/a
        return Loss2/new_y.size

def comp():
    #test_x = torch.unsqueeze(torch.from_numpy(x[10]),dim=1)
    #test_y = torch.unsqueeze(torch.from_numpy(y[65]),dim=1)
    test_x = pow(10,x[10])
    test_y = pow(10,y[100])
    loss=0
    for i,j in zip(test_x,test_y):
        loss += abs(i-j)/j
    return loss/test_x.size

ans = 0

if(args.method == 0):
    for cnt in range(2):
        ans+=test(cnt)
    ans/=2
elif(args.method == 1):
    train(1200)
    #for cnt in range(2):
     #   ans+=test(cnt)
    ans = test(1)

else:
    ans = comp()

file = open('result.txt','a')
file.write(str(1-ans))
file.write('\n')
if(args.event_id == 12):
    file.write('=============================\n')
file.close()
