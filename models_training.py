import torch
import torch.nn as nn
import torch.nn.functional as F
from random import shuffle
import pandas as pd
from pathlib import Path
import os
from vad_funcs import train_until_test_is_not_improving,evaluate_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

""" first lets test a basic logistic regression """

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

print('done with linear regression, starting naive net')
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

hidden_size = input_size * 2
model = NaiveNet(input_size,hidden_size,number_of_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

training_df,trained_model = \
    train_until_test_is_not_improving(device,model,criterion,optimizer,150)

training_df.to_csv(r'models_results/'+f'{model_name}_training.csv',index=False)

evaluate_model(model, device, model_name)

###########
#rnn
model_name = 'rnn'


class Rnn(torch.nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size):
        super(Rnn, self).__init__()
        self.embedding = torch.nn.Embedding(input_size, embedding_size)
        self.rnn = torch.nn.LSTM(embedding_size, hidden_size)
        self.out = torch.nn.Linear(hidden_size, output_size)
        self.v = torch.nn.Parameter(torch.FloatTensor(hidden_size).uniform_(-0.2, 0.2)).cuda()
        self.fc = torch.nn.Linear(hidden_size, hidden_size)

    def forward(self, single_sentence):
        embedded = self.embedding(single_sentence)
        embedded = embedded.view(len(single_sentence), 1, -1)
        out, hidden = self.rnn(embedded)

        x = self.fc(torch.squeeze(out))
        x = torch.tanh(x)
        probobilty = torch.matmul(x, self.v)
        weights = F.softmax(probobilty, dim=0)
        context = torch.matmul(weights.T, torch.squeeze(out))
        return self.out(context)

# #rnn
# model_name = 'bi_rnn'
#
# #rnn
# model_name = 'lstm'
#
# #rnn
# model_name = 'bi_lstm'