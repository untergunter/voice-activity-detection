import torch
import torch.nn as nn
import torch.nn.functional as F
from random import shuffle
import pandas as pd
import glob
from pathlib import Path
import os
from typing import Collection

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'using {device}')


def train_and_test_files_paths(path: str = os.getcwd(), test_ratio: float = 0.004) -> list:
    files = [file_name for file_name in Path(os.getcwd()).rglob("*.gzip")]
    shuffle(files)
    first_train = int(test_ratio * len(files))
    test = files[:first_train]
    train = files[first_train:]
    return train, test


def df_to_X_y(path):
    df = pd.read_parquet(path)
    y_column = df.columns[-1]
    X_columns = [col for col in df.columns if not col == y_column]
    X = torch.Tensor(df[X_columns].values)
    y = torch.from_numpy(df[y_column].values)
    y = y.type(torch.LongTensor)
    return X, y


class ParquetLoader():
    def __init__(self, files: list, meta_data: pd.DataFrame = None):
        self.file_names = files
        if meta_data is None:
            self.meta_data = None
        else:
            self.meta_data = meta_data


    def shuffle_inputs(self):
        shuffle(self.file_names)

    def __iter__(self):
        self.shuffle_inputs()
        for file in self.file_names:
            yield df_to_X_y(file)

    def __len__(self):
        return len(self.file_names)


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
    files = [file_name for file_name in Path(os.getcwd()).rglob("*.gzip")][:100]
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
    snr_list, noise_list, number_of_rows, number_of_talk, file_numbers = [], [], [], [], []
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
        file_numbers.append(file_name)
    files_df = pd.DataFrame({"path": raw_file_names
                            , "noise": noise_list
                            , "snr": snr_list
                            , "rows": number_of_rows
                            , "rows_of_speech": number_of_talk
                            , "file_number": file_name
                             })
    return files_df

def make_path_list_and_metadata(paths:set,full_df):
    meta_data = full_df[full_df['file_number'].isin(paths)]
    meta_data = meta_data.set_index('path').T.to_dict('list')
    paths = list(paths)
    return paths,meta_data

def train_test_validate_split(train_ratio: float = 0.95, test_ratio: float = 0.04) -> tuple:
    """ this function returns 3 data loaders """
    assert train_ratio + test_ratio < 1, 'we must have a validation set,' \
                                         ' so train_ratio+test_ratio must be <1'
    all_data_files = find_all_data_files()
    distinct_files = list(all_data_files['file_number'].unique())
    shuffle(distinct_files)
    first_train = int(train_ratio * len(distinct_files))
    first_validation = int(test_ratio + train_ratio * len(distinct_files))
    train_names = set(distinct_files[:first_train])
    test_names = set(distinct_files[first_train:first_validation])
    validation_names = set(distinct_files[first_validation:])
    all_loaders = [ParquetLoader(make_path_list_and_metadata(set_of_paths,all_data_files))
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
