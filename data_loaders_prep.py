from vad_funcs import train_test_validate_split
import pickle
import pathlib

folder = pathlib.Path("data_loaders/")
print(folder.resolve())
folder.mkdir(exist_ok=True)
loaders = train_test_validate_split()
for name, loader_object in zip(('train', 'test', 'validate'), loaders):
    file = folder / f'{name}.pickle'
    print(file.resolve())
    with file.open("wb") as output_file:
        pickle.dump(loader_object, output_file)