import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
import numpy as np

from sklearn.model_selection import train_test_split

t_loan = pd.read_csv('loan_11_train.csv').drop('uid',axis=1)

# t_loan.drop('label',axis=1,inplace=True)
# t_loan.groupby(level=0,axis=1).apply(lambda x:print(x.count()))
# print(t_loan.astype(bool).sum(axis=0))
# print()
'''
input: loan，exp(loan), period
最高68个借贷数据，利用11月数据拟合总数据

'''

class LoanDataset(torch.utils.data.Dataset):
    """Face Landmarks dataset."""

    def __init__(self, df, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx, :]
        data = torch.from_numpy(np.array(row.drop('label'))).float()
        label = torch.FloatTensor([row['label']])
        sample = {'data': data, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample


class Net(nn.Module):

    def __init__(self):
        super(Net,self).__init__()
        self.fc1 = nn.Linear(204,200)
        self.fc2 = nn.Linear(200,100)
        self.fc3 = nn.Linear(100,50)
        self.fc4 = nn.Linear(50, 5)
        self.fc5 = nn.Linear(5, 1)

    def forward(self,x):
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = F.tanh(self.fc3(x))
        x = F.tanh(self.fc4(x))
        x = F.tanh(self.fc5(x))
        return x

    # def num_flat_features(self, x):
    #     size = x.size()[1:]  # all dimensions except the batch dimension
    #     num_features = 1
    #     for s in size:
    #         num_features *= s
    #     # print('num_features: %d' % num_features)
    #     return num_features

    # def num_flat_features(self,x):
    #     size = x.size()[1:]
    #     return reduce(lambda x,y:x*y,size,1)

net = torch.load('mytraining.pt')
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
criterion = nn.MSELoss()
# zero the parameter gradients
# optimizer.zero_grad()

for epoch in range(200):  # loop over the dataset multiple times

    train, test = train_test_split(t_loan, test_size=0.4)
    train_dataset = LoanDataset(train)
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=4, num_workers=2)
    test_dataset = LoanDataset(test)
    testloader = torch.utils.data.DataLoader(train_dataset, batch_size=4, num_workers=2)

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs = data['data']
        labels = data['label']

        # wrap them in Variable
        inputs, labels = Variable(inputs), Variable(labels)

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.data[0]
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] train_loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

    test_loss = 0.0
    for i, data in enumerate(testloader, 0):
        # get the inputs
        inputs = data['data']
        labels = data['label']
        # wrap them in Variable
        inputs, labels = Variable(inputs), Variable(labels)
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        # print statistics
        test_loss += loss.data[0]
        if i == 500:
            break
    print('[%d, %5d] test_loss: %.3f' % (epoch + 1, i + 1, test_loss / 500))

print('Finished Training')

# ... after training, save your model
# net.save_state_dict('mytraining.pt')
torch.save(net,'mytraining.pt')
# .. to load your previously training model:
# net.load_state_dict(torch.load('mytraining.pt'))