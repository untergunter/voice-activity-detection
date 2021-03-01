import torch
import torch.nn as nn
import torch.nn.functional as F
from random import shuffle
import pandas as pd
from pathlib import Path
import os
from vad_funcs import train_until_test_is_not_improving,evaluate_model\
    ,batch_train_until_test_is_not_improving,batch_evaluate_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

""" first lets test a basic logistic regression """

def logistic_regression_train(device):
    input_size = 26
    number_of_classes = 2
    learning_rate = 0.001

    model_name = 'logistic_regression'
    class LogisticRegression(torch.nn.Module):
        def __init__(self, input_size, number_of_classes):
            super(LogisticRegression, self).__init__()
            self.linear = torch.nn.Linear(input_size, number_of_classes)

        def forward(self, x):
            x = self.linear(x)
            x = F.log_softmax(x, dim=1)
            return x

    model = LogisticRegression(input_size,number_of_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    training_df,trained_model = \
        train_until_test_is_not_improving(device,model,criterion,optimizer,150)

    training_df.to_csv(r'models_results/'+f'{model_name}_training.csv',index=False)

    evaluate_model(model, device, model_name)


def naive_net_train(device):
    model_name = 'naive_net'
    class NaiveNet(nn.Module):

        def __init__(self, input_size, hidden_size, number_of_classes):
            super(NaiveNet, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, number_of_classes)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    input_size = 26
    number_of_classes = 2
    learning_rate = 0.001
    hidden_size = input_size * 2
    model = NaiveNet(input_size,hidden_size,number_of_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    training_df,trained_model = \
        train_until_test_is_not_improving(device,model,criterion,optimizer,150)

    training_df.to_csv(r'models_results/'+f'{model_name}_training.csv',index=False)

    evaluate_model(model, device, model_name)

###########

# changed data loading to load all noise permutatinos of each
# file together so train is now every iteration is 31 taged files

def batch_logistic_regression_train(device):
    input_size = 26
    number_of_classes = 2
    learning_rate = 0.001

    model_name = 'logistic_regression'
    class LogisticRegression(torch.nn.Module):
        def __init__(self, input_size, number_of_classes):
            super(LogisticRegression, self).__init__()
            self.linear = torch.nn.Linear(input_size, number_of_classes)

        def forward(self, x):
            x = self.linear(x)
            x = F.log_softmax(x, dim=1)
            return x

    model = LogisticRegression(input_size,number_of_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    training_df,trained_model = \
        batch_train_until_test_is_not_improving(device,model,criterion,optimizer,stop_after_not_improving_for=20)

    training_df.to_csv(r'models_results/'+f'{model_name}_training.csv',index=False)

    batch_evaluate_model(model, device, model_name)


def batch_naive_net_train(device):
    model_name = 'naive_net'

    class NaiveNet(nn.Module):

        def __init__(self, input_size, hidden_size, number_of_classes):
            super(NaiveNet, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, number_of_classes)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    input_size = 26
    number_of_classes = 2
    learning_rate = 0.001
    hidden_size = input_size * 2
    model = NaiveNet(input_size,hidden_size,number_of_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    training_df,trained_model = \
        batch_train_until_test_is_not_improving(device,model,criterion,optimizer,stop_after_not_improving_for=25)

    training_df.to_csv(r'models_results/'+f'{model_name}_training.csv',index=False)

    batch_evaluate_model(model, device, model_name)
if __name__=='__main__':
    batch_naive_net_train(device)