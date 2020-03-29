import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import torch

torch.set_default_dtype(torch.float32)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.hidden_layer = nn.Linear(4,4,bias=True)
        self.final_layer = nn.Linear(4,3,bias=True)
        self.activation_fn = lambda x: F.softmax(x, dim=1)
    
    def forward(self, x):
        x = self.activation_fn(self.hidden_layer(x))
        x = self.activation_fn(self.final_layer(x))
        return x

    def train(self, dataset):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.parameters(), lr=0.01)
        loader = DataLoader(dataset, batch_size=4,shuffle=True,num_workers=2)


        for epoch in range(1000):
            for data in loader:
                X_batch, y_batch = data
                optimizer.zero_grad()
                y_hat = self.forward(X_batch)
                loss = criterion(y_hat, y_batch)
                loss.backward()
                optimizer.step()

        print("Finished Training")
    
    def test(self, dataset):
         loader = DataLoader(dataset, batch_size=4,shuffle=True,num_workers=2)
         correct, total = 0, 0
         with torch.no_grad():
            for data in loader:
                X_batch, y_batch = data
                y_hat = self.forward(X_batch)
                MAP_estimators = torch.max(y_hat,1)[1]
                total += y_batch.size(0)
                print (MAP_estimators)
                print(y_batch)
                correct += (MAP_estimators == y_batch).sum().item()
         print("Accuracy: ", 100*(correct/total))




dataset = pd.read_csv("./IRIS.csv")

for species, enum in [('Iris-versicolor', 0),('Iris-setosa', 1),('Iris-virginica',2)]:
    dataset.loc[dataset.species==species, 'species'] = enum

dataset = dataset.sample(frac=1)

X_train = torch.tensor(dataset[dataset.columns[:-1]][0:100].values,dtype=torch.float32)
y_train = torch.tensor(dataset[dataset.columns[-1]][0:100].values,dtype=torch.long)
X_test  = torch.tensor(dataset[dataset.columns[:-1]][100:150].values,dtype=torch.float32)
y_test  = torch.tensor(dataset[dataset.columns[-1]][100:150].values,dtype=torch.long)
 

training_dataset = TensorDataset(X_train,y_train)
testing_dataset = TensorDataset(X_test,y_test)


model = Model()
model.train(training_dataset)
model.test(testing_dataset)
