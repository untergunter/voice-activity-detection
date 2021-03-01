import torch
import torch.nn as nn
import torch.nn.functional as F
from random import shuffle
import pandas as pd
from pathlib import Path
import os
import pickle
import numpy as np
from collections import defaultdict


def train_and_test_files_paths(path: str = os.getcwd(), test_ratio: float = 0.004) -> list:
    files = [file_name for file_name in Path(os.getcwd()).rglob("*.gzip")]
    shuffle(files)
    first_train = int(test_ratio * len(files))
    test = files[:first_train]
    train = files[first_train:]
    return train, test


def df_path_to_X_y(path):
    df = pd.read_parquet(path)
    y_column = df.columns[-1]
    X_columns = [col for col in df.columns if not col == y_column]
    X = torch.Tensor(df[X_columns].values)
    y = torch.from_numpy(df[y_column].values)
    y = y.type(torch.LongTensor)
    return X, y


class ParquetLoader:

    def __init__(self, files: list, meta_data: dict = None):
        self.file_names = files
        self.index = 0
        if meta_data is None:
            self.meta_data = None
        else:
            self.meta_data = meta_data

    def shuffle_inputs(self):
        shuffle(self.file_names)

    def __iter__(self):
        self.shuffle_inputs()
        for file in self.file_names:
            yield df_path_to_X_y(file)

    def __len__(self):
        return len(self.file_names)

    def __next__(self):
        try:
            result = self.file_names[self.index] if self.index < len(self) else None
        except IndexError:
            result = None
        self.index += 1
        return result


class NaiveNet(nn.Module):

    def __init__(self, input_size, hidden_size, number_of_classes):
        super(NaiveNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, number_of_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def get_original_name(file_name):
    return file_name.parts[-1]


def get_all_files():
    files = [file_name for file_name in Path(os.getcwd()).rglob("*.gzip")]
    return files


def split_to_train_test(files, test_ratio):
    distinct_files = list({get_original_name(file) for file in files})
    print(len(distinct_files))
    shuffle(distinct_files)
    first_train = int(test_ratio * len(distinct_files))
    test_names = set(files[:first_train])
    train = test = list()
    for file in files:
        if get_original_name(file) in test_names:
            test.append(file)
        else:
            train.append(file)
    return train, test


def get_noise_type_and_snr(file):
    snr = ''.join(charecter for charecter in file.parts[-2] if charecter.isnumeric())
    if snr == '':
        noise_type = 'Quiet'
        snr = 0
    elif snr[:2] == '80':
        noise_type = 'car80'
        snr = snr[2:]
    else:
        noise_type = file.parts[-3]
    return noise_type, snr


def find_all_data_files() -> pd.DataFrame:
    raw_file_names = get_all_files()
    snr_list, noise_list, number_of_rows, number_of_talk, file_namess = [], [], [], [], []
    for raw_file in raw_file_names:
        noise, snr = get_noise_type_and_snr(raw_file)
        noise_list.append(noise)
        snr_list.append(snr)

        df = pd.read_parquet(raw_file)
        n_rows = df.shape[0]
        number_of_talk_rows = df[df.columns[-1]].sum().astype(int)

        number_of_rows.append(n_rows)
        number_of_talk.append(number_of_talk_rows)

        file_name = get_original_name(raw_file)
        file_namess.append(file_name)
    files_df = pd.DataFrame({"path": raw_file_names
                                , "noise": noise_list
                                , "snr": snr_list
                                , "rows": number_of_rows
                                , "rows_of_speech": number_of_talk
                                , "file_number": file_namess
                             })
    return files_df


def make_path_list_and_metadata(file_numbers: set, full_df):
    meta_data = full_df[full_df['file_number'].isin(file_numbers)]
    paths = meta_data['path'].to_list()
    meta_data = meta_data.set_index('path').to_dict('index')
    return paths, meta_data


def train_test_validate_split(train_ratio: float = 0.95, test_ratio: float = 0.04) -> tuple:
    """ this function returns 3 data loaders """
    assert train_ratio + test_ratio < 1, 'we must have a validation set,' \
                                         ' so train_ratio+test_ratio must be <1'
    all_data_files = find_all_data_files()
    distinct_files = list(all_data_files['file_number'].unique())
    shuffle(distinct_files)
    first_train = int(train_ratio * len(distinct_files))
    first_validation = int((test_ratio + train_ratio) * len(distinct_files))
    train_names = set(distinct_files[:first_train])
    test_names = set(distinct_files[first_train:first_validation])
    validation_names = set(distinct_files[first_validation:])

    all_loaders = [ParquetLoader(*make_path_list_and_metadata(set_of_paths, all_data_files))
                   for set_of_paths in (train_names, test_names, validation_names)]
    return all_loaders


class NaiveNet(nn.Module):

    def __init__(self, input_size, hidden_size, number_of_classes):
        super(NaiveNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, number_of_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def load_pickle(path):
    with open(path, "rb") as input_file:
        result = pickle.load(input_file)
    return result


def train_until_test_is_not_improving(device
                                      , model
                                      , criterion
                                      , optimizer
                                      , stop_after_not_improving_for: int = 200):
    train_loader = load_pickle(r'data_loaders/train.pickle')
    test_loader = load_pickle(r'data_loaders/test.pickle')

    model_didnt_improve_for = 0
    losses = []
    accuracy_history = []
    best_accuracy = 0

    while stop_after_not_improving_for > model_didnt_improve_for:

        next_one = train_loader.__next__()
        if next_one is None:  # we finished all the data
            train_loader.shuffle_inputs()
            next_one = train_loader.__next__()
        X, y = df_path_to_X_y(next_one)
        X = X.to(device)
        y = y.to(device)

        # foreword
        output = model(X)
        loss = criterion(output, y)
        losses.append(loss)

        # backwards
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # check our early stop condition - is accuracy getting better?
        with torch.no_grad():
            next_one = test_loader.__next__()
            if next_one is None:  # we finished all the data
                test_loader.shuffle_inputs()
                next_one = test_loader.__next__()
            X, y = df_path_to_X_y(next_one)
            X = X.to(device)
            y = y.to(device)

            # foreword
            outputs = model(X)
            _, predictions = torch.max(outputs, 1)

            # calculate accuracy
            total = predictions.shape[0]
            correct = (predictions == y).sum().item()
            accuracy = 100 * correct / total
            accuracy_history.append(accuracy)

            # is accuracy getting better?
            last_30_mean = np.mean(accuracy_history[-30:])
            if best_accuracy >= last_30_mean:
                model_didnt_improve_for += 1
            elif best_accuracy < last_30_mean:
                model_didnt_improve_for = 0
                best_accuracy = last_30_mean

            if len(accuracy_history) % 100 == 0:
                print(f'last loss is {loss},'
                      f'accuracy is {accuracy},'
                      f'mean accuracy is {last_30_mean},'
                      f'didnt improve for {model_didnt_improve_for},')
    loss_accuracy_df = pd.DataFrame({'loss': torch.tensor(losses).to('cpu')
                                        , 'accuracy': torch.tensor(accuracy_history).to('cpu')})
    return loss_accuracy_df, model


def evaluate_model(model, device, model_name):
    validation_set = load_pickle(r'data_loaders/validate.pickle')
    model_results = validation_set.meta_data.copy()
    while True:
        with torch.no_grad():
            next_one = validation_set.__next__()
            if next_one is None:  # we finished all the data
                break
            X, y = df_path_to_X_y(next_one)
            X = X.to(device)
            y = y.to(device)

            # foreword
            outputs = model(X)
            _, predictions = torch.max(outputs, 1)

            # calculate accuracy
            total = predictions.shape[0]
            correct = (predictions == y).sum().item()
            accuracy = 100 * correct / total

            # save_results
            model_results[next_one]['accuracy'] = accuracy

    # write results to dataframe
    model_results = pd.DataFrame(model_results).T
    model_results.reset_index(drop=True, inplace=True)
    model_results.sort_values(by='accuracy', ascending=False, inplace=True)
    model_results.to_csv(r'models_results/' + f'{model_name}.csv', index=False)


class BatchParquetLoader:

    def __init__(self, base_files: list, augmentations_paths_per_base_file: dict = None):
        self.base_files = list(base_files)
        self.augmentation_paths_per_base_file = augmentations_paths_per_base_file
        self.index = 0

    def shuffle_inputs(self):
        shuffle(self.base_files)

    def get_all_augmentations(self,file):
        all_file_augmentations = [df_path_to_X_y(specific_augmentation)
                                  for specific_augmentation in self.augmentation_paths_per_base_file[file]]
        return all_file_augmentations

    def get_all_paths_names(self,file):
        all_paths = [specific_augmentation for specific_augmentation in self.augmentation_paths_per_base_file[file]]
        return all_paths

    def __iter__(self):
        for file in self.base_files:
            all_file_augmentations = self.get_all_augmentations(file)
            yield all_file_augmentations

    def __len__(self):
        return len(self.base_files)

    def __next__(self):
        try:
            next_file = self.base_files[self.index] if self.index < len(self) else None

        except IndexError:
            self.index = 0
            self.shuffle_inputs()
            next_file = self.base_files[self.index] if self.index < len(self) else None

        self.index += 1
        return next_file


def build_set_of_base_files_and_paths_dict():
    all_paths = get_all_files()

    paths_for_base_file = defaultdict(list)
    distinct_base_files = set()

    for specific_path in all_paths:
        base_file = get_original_name(specific_path)
        paths_for_base_file[base_file].append(specific_path)
        distinct_base_files.add(base_file)
    return distinct_base_files, paths_for_base_file


def train_test_validate_split_bp(train_ratio: float = 0.95, test_ratio: float = 0.04) -> tuple:
    """ this function returns 3 data loaders """
    assert train_ratio + test_ratio < 1, 'we must have a validation set,' \
                                         ' so train_ratio+test_ratio must be <1'

    base_files, paths_per_base_file = build_set_of_base_files_and_paths_dict()

    base_files = list(base_files)
    shuffle(base_files)

    #find indexes
    first_train = int(train_ratio * len(base_files))
    first_validation = int((test_ratio + train_ratio) * len(base_files))

    #split by indexes
    train_names = set(base_files[:first_train])
    test_names = set(base_files[first_train:first_validation])
    validation_names = set(base_files[first_validation:])

    #prepare loaders
    all_loaders = [BatchParquetLoader(base_files_subset
                                      , {base_file: paths_per_base_file[base_file]
                                         for base_file in base_files_subset})
                   for base_files_subset in (train_names, test_names, validation_names)]
    return all_loaders

def batch_train_until_test_is_not_improving(device
                                      , model
                                      , criterion
                                      , optimizer
                                      , stop_after_not_improving_for: int = 5):
    train_loader = load_pickle(r'data_loaders/trainb.pickle')
    test_loader = load_pickle(r'data_loaders/testb.pickle')

    model_didnt_improve_for = 0
    losses = []
    accuracy_history = []
    best_accuracy = 0

    while stop_after_not_improving_for > model_didnt_improve_for:
        for next_batch in train_loader:
            batch_loss = []
            for X,y in next_batch:
                X = X.to(device)
                y = y.to(device)

                # foreword

                optimizer.zero_grad()
                output = model(X)
                loss = criterion(output, y)


            # backwards
                loss.backward()
                optimizer.step()

                batch_loss.append(loss)

            # save batch loss
            losses.append(torch.mean(torch.tensor(batch_loss)).item())
            # check our early stop condition - is accuracy getting better?
            with torch.no_grad():
                next_file = test_loader.__next__()
                if next_file is None:  # we finished all the data
                    test_loader.shuffle_inputs()
                    test_loader.index=0
                    next_file = test_loader.__next__()

                next_batch = test_loader.get_all_augmentations(next_file)

                batch_accuracy = []

                for X, y in next_batch:

                    X = X.to(device)
                    y = y.to(device)


                    # foreword
                    outputs = model(X)
                    _, predictions = torch.max(outputs, 1)

                # calculate accuracy
                    total = predictions.shape[0]
                    correct = (predictions == y).sum().item()
                    accuracy = 100 * correct / total
                    batch_accuracy.append(accuracy)

                last_batch_accuracy = torch.mean(torch.tensor(batch_accuracy)).item()
                accuracy_history.append(last_batch_accuracy)
                # is accuracy getting better

                if best_accuracy >= last_batch_accuracy:
                    model_didnt_improve_for += 1
                elif best_accuracy < last_batch_accuracy:
                    model_didnt_improve_for = 0
                    best_accuracy = last_batch_accuracy

                print(f'last batch loss is {losses[-1]},'
                      f'accuracy is {last_batch_accuracy},'
                      f'didnt improve for {model_didnt_improve_for},')

            if stop_after_not_improving_for <= model_didnt_improve_for:
                break
    loss_accuracy_df = pd.DataFrame({'loss': torch.tensor(losses).to('cpu')
                                        , 'accuracy': torch.tensor(accuracy_history).to('cpu')})
    return loss_accuracy_df, model

def batch_evaluate_model(model, device, model_name):
    validation_set = load_pickle(r'data_loaders/validateb.pickle')
    results = []

    with torch.no_grad():
        for index in range(len(validation_set)):
            base_name = validation_set.base_files[index]
            augmentations = validation_set.get_all_paths_names(base_name)
            for path in augmentations:
                X,y = df_path_to_X_y(path)
                X = X.to(device)
                y = y.to(device)

                # foreword
                outputs = model(X)
                _, predictions = torch.max(outputs, 1)

                # calculate accuracy
                total = predictions.shape[0]
                correct = (predictions == y).sum().item()
                accuracy = 100 * correct / total

                # save_results
                noise,snr = get_noise_type_and_snr(path)
                results.append((base_name,noise,snr,accuracy))

    # write results to dataframe
    model_results = pd.DataFrame(results,columns=['base_name','noise','snr','accuracy'])
    model_results.sort_values(by='accuracy', ascending=False, inplace=True)
    model_results.to_csv(r'models_results/' + f'{model_name}.csv', index=False)