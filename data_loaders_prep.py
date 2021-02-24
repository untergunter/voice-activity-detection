import pathlib
from vad_funcs import train_test_validate_split
import pickle

folder = pathlib.Path("data_loaders/")
folder.mkdir(exist_ok=True)
loaders = train_test_validate_split()
for name, loader_object in zip(('train', 'test', 'validate'), loaders):
    file = folder / f'{name}.pickle'
    print(file.resolve())
    with file.open("wb") as output_file:
        pickle.dump(loader_object, output_file)


with open('/home/ido/data/idc/deep learning/vad/data_loaders/train.pickle', 'rb') as f:
    tr = pickle.load(f)

with open('/home/ido/data/idc/deep learning/vad/data_loaders/test.pickle', 'rb') as f:
    ts = pickle.load(f)

with open('/home/ido/data/idc/deep learning/vad/data_loaders/validate.pickle', 'rb') as f:
    vl = pickle.load(f)

for dataloader in (tr,ts,vl):
    print(len(dataloader))
    if dataloader.meta_data is not None:
        print(dataloader.meta_data)