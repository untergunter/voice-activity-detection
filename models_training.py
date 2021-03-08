import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
import pandas as pd
from vad_funcs import \
    train_until_test_is_not_improving,evaluate_model\
    ,batch_train_until_test_is_not_improving,batch_evaluate_model\
    ,batch_train_rnn_until_test_is_not_improving,batch_evaluate_rnn_model

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

# changed data loading to load all noise permutations of each
# file together so train is now every iteration is 31 taged files

def batch_logistic_regression_train(device):
    input_size = 26
    number_of_classes = 2
    learning_rate = 0.001

    model_name = 'batch_logistic_regression'
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
    model_name = 'batch_naive_net'

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

def batch_deep_neural_net_train(device):
    model_name = 'deep_neural_net'

    class DeepNeuralNet(nn.Module):

        def __init__(self, input_size, hidden_size, number_of_classes):
            super(DeepNeuralNet, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size*4)
            self.fc2 = nn.Linear(hidden_size*4, hidden_size*3)
            self.fc3 = nn.Linear(hidden_size*3, hidden_size*2)
            self.fc4 = nn.Linear(hidden_size*2, hidden_size)
            self.fc5 = nn.Linear(hidden_size,number_of_classes)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            x = F.relu(self.fc4(x))
            x = self.fc5(x)
            return x

    input_size = 26
    number_of_classes = 2
    learning_rate = 0.001
    hidden_size = input_size * 2
    model = DeepNeuralNet(input_size,hidden_size,number_of_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    training_df,trained_model = \
        batch_train_until_test_is_not_improving(device,model,criterion,optimizer,stop_after_not_improving_for=25)

    training_df.to_csv(r'models_results/'+f'{model_name}_training.csv',index=False)

    batch_evaluate_model(model, device, model_name)


def batch_rnn_train(device):
    model_name = 'batch_rnn'

    class RNN(nn.Module):

        def __init__(self, input_size, hidden_size, number_of_classes):
            self.hidden_size = hidden_size
            super(RNN, self).__init__()

            # make new hidden layer
            self.fc1 = nn.Linear(input_size + hidden_size,  input_size + hidden_size)
            self.fc2 = nn.Linear(input_size + hidden_size, hidden_size)

            # make output
            self.fc3 = nn.Linear(input_size + hidden_size, input_size + hidden_size)
            self.fc4 = nn.Linear(input_size + hidden_size, number_of_classes)

        def forward(self, x,hidden_vec):

            full_input = torch.cat((x,hidden_vec))
            new_hidden_vec = F.leaky_relu(self.fc1(full_input))
            new_hidden_vec = self.fc2(new_hidden_vec)

            output = F.leaky_relu(self.fc3(full_input))
            output = self.fc4(output)
            return output, new_hidden_vec

        def init_hidden(self):
            return torch.rand(self.hidden_size)


    input_size = 26
    number_of_classes = 2
    learning_rate = 0.001
    hidden_size = input_size * 2
    model = RNN(input_size,hidden_size,number_of_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    training_df,trained_model = \
        batch_train_rnn_until_test_is_not_improving(device,model,criterion,optimizer,stop_after_not_improving_for=25)

    training_df.to_csv(r'models_results/'+f'{model_name}_training.csv',index=False)

    batch_evaluate_rnn_model(model, device, model_name)


def batch_deep_neural_net_no_delta_train(device):
    model_name = 'atch_deep_neural_net_no_delta'

    class DeepNaiveNet(nn.Module):

        def __init__(self, input_size, hidden_size, number_of_classes):
            super(DeepNaiveNet, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size*4)
            self.fc2 = nn.Linear(hidden_size*4, hidden_size*3)
            self.fc3 = nn.Linear(hidden_size*3, hidden_size*2)
            self.fc4 = nn.Linear(hidden_size*2, hidden_size)
            self.fc5 = nn.Linear(hidden_size,number_of_classes)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            x = F.relu(self.fc4(x))
            x = self.fc5(x)
            return x

    input_size = 10
    number_of_classes = 2
    learning_rate = 0.001
    hidden_size = input_size * 2
    model = DeepNaiveNet(input_size,hidden_size,number_of_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    training_df,trained_model = \
        batch_train_until_test_is_not_improving(device,model,criterion,optimizer
                                                ,stop_after_not_improving_for=25,no_delta=True)

    training_df.to_csv(r'models_results/'+f'{model_name}_training.csv',index=False)

    batch_evaluate_model(model, device, model_name,no_delta = True)


def batch_rnn_no_delta_train(device):
    model_name = 'batch_rnn_no_delta'

    class RNN(nn.Module):

        def __init__(self, input_size, hidden_size, number_of_classes):
            self.hidden_size = hidden_size
            super(RNN, self).__init__()

            # make new hidden layer
            self.fc1 = nn.Linear(input_size + hidden_size,  input_size + hidden_size)
            self.fc2 = nn.Linear(input_size + hidden_size, hidden_size)

            # make output
            self.fc3 = nn.Linear(input_size + hidden_size, input_size + hidden_size)
            self.fc4 = nn.Linear(input_size + hidden_size, number_of_classes)

        def forward(self, x,hidden_vec):

            full_input = torch.cat((x,hidden_vec))
            new_hidden_vec = F.leaky_relu(self.fc1(full_input))
            new_hidden_vec = self.fc2(new_hidden_vec)

            output = F.leaky_relu(self.fc3(full_input))
            output = self.fc4(output)
            return output, new_hidden_vec

        def init_hidden(self):
            return torch.rand(self.hidden_size)


    input_size = 10
    number_of_classes = 2
    learning_rate = 0.001
    hidden_size = input_size * 2
    model = RNN(input_size,hidden_size,number_of_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    training_df,trained_model = \
        batch_train_rnn_until_test_is_not_improving(device,model,criterion,optimizer
                                                    ,stop_after_not_improving_for=25,no_delta=True)

    training_df.to_csv(r'models_results/'+f'{model_name}_training.csv',index=False)

    batch_evaluate_rnn_model(model, device, model_name,no_delta = True)

if __name__=='__main__':
    model_name = ['logistic_regression','naive_net','batch_logistic_regression'
                 ,'batch_naive_net' ,'batch_deep_neural_net','batch_rnn'
                 ,'batch_deep_neural_net_no_delta','batch_rnn_no_delta']
    time_to_train = []
    for training_function in(logistic_regression_train,naive_net_train,batch_logistic_regression_train
                            ,batch_naive_net_train,batch_deep_neural_net_train,batch_rnn_train
                            ,batch_deep_neural_net_no_delta_train,batch_rnn_no_delta_train):
        t0 = datetime.now()
        training_function(device)
        t1 = datetime.now()
        time_to_train.append((t1-t0).seconds)

    training_time = pd.DataFrame({'model': model_name, 'time': time_to_train})
    training_time.sort_values(by='time', ascending=False, inplace=True)
    training_time.to_csv('training_time.csv', index=False)