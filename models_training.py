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

model = LogisticRegression()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

training_df,trained_model = train_until_test_is_not_improving(
                            device,model,criterion,optimizer):

training_df.to_parquet(f'{model_name}_training.parquet',index=False)

evaluate_model(model, device, model_name)