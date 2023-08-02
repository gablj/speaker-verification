import os
import shutil
import random

def train_test_split(root_dir: str, dir_train: str, dir_test: str, train_split: float = 0.75) -> None:
    """
    Splits the previously preprocessed data (see 'preprocess_data.py') mel-filterbank energies
    numpy arrays of floats into train and test directories, which contain subdirectories 
    for each speaker. In addition, each speaker subdirectory has a txt file 'frames.txt'
    that has a list of all the mel-filterbank energies numpy arrays names from the 
    corresponding speaker that are present on the subdirectory. The resulting 
    split directories have the following scheme: 
    .- train/
        |
            .- subdir_speaker_idx/
                |
                _frames.txt
                sample_idx_xxxx.npy
                sample_idx_xxxy.npy
                ...
            .- subdir_speaker_idy/
                |
                ...
            ...
    Parameters
    ----------
    root_dir : str
        String path where the previously preprocessed speakers directories 
        containing the mel-filterbank energies as numpy arrays of floats
        are located. 
    dir_train : str 
        String path where the train split is going to be stored. 
    dir_test : str 
        String path where the test split is going to be stored.
    train_split : float, default = 0.75 
        The fraction out of the complete samples that will be used
        for training.
    """
    # Check if a previous train split exists, if so, it deletes it 
    if os.listdir(dir_train):
        shutil.rmtree(dir_train) #To delete the subdirectory and all its contens 
        print("A previous train split was present at '%s', it has been removed " % dir_train)
    # Check if a previous test split exists, if so, it deletes it 
    if os.listdir(dir_test):
        shutil.rmtree(dir_test) #To delete the subdirectory and all its contens 
        print("A previous test split was present at '%s', it has been removed " % dir_test)
    # Get the list of subdirectories in the main directory
    # Filters only the subdirectories names 'LibriSpeech_train-other-500_XXXX'
    subdirectories = [subdir for subdir in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, subdir))]

    # Randomly shuffle the list of subdirectories
    random.shuffle(subdirectories)

    # Calculate the number of directories for train and test
    train_count = int(len(subdirectories) * train_split)

    # Split the subdirectories into train and test sets
    train_subdir = subdirectories[:train_count]
    test_subdir = subdirectories[train_count:]

    # Create the train and test directories if they don't exist 
    # or were deleated above 
    os.makedirs(dir_train, exist_ok=True)
    os.makedirs(dir_test, exist_ok=True)

    # Copy directories from the preprocessed directory to the train directory
    for subdir in train_subdir:
        shutil.copytree(os.path.join(root_dir, subdir), os.path.join(dir_train, subdir), dirs_exist_ok=True)
    # Adding a .txt file to each subdirectory, that stores the
    # '.npy, target' contained in the corresponding subdirectory of the train split 
    _add_frames(dir_train)


    # Copy directories from the preprocessed directory to the test directory
    for subdir in test_subdir:
        shutil.copytree(os.path.join(root_dir, subdir), os.path.join(dir_test, subdir))
    # Adding a .txt file to each subdirectory, that stores the
    # '.npy, target' contained in the corresponding subdirectory of the test split 
    _add_frames(dir_test)

    print("Split completed succesfully.")

    # Verify that train_subdir and test_subdir are disjoint
    train_set = set(train_subdir)
    test_set = set(test_subdir)

    if train_set.isdisjoint(test_set):
        print("The train and test splits are disjoint.")
    else:
        print("The train and test splits are not disjoint.")


def _add_frames(dir_path : str) -> None: 
    """
    "Private" function (not intended to be accessed outside the module or by users) that 
    adds a txt file that, for each subdirectory of the train/test split, 
    that lists all '.npy' files present in the correspoding subdirectory as 
    '.npy, target_speaker'.
    ....
    Parameters
    ----------
    dir_path : str 
        A string of the path of the train or test directory   
    """
    for subdir in os.listdir(dir_path):
        subdir_path = os.path.join(dir_path, subdir)
        npy_files = []
        target_speaker_id = subdir.split('_')[-1] #To get the speaker id
        for file in os.listdir(subdir_path):
            if file.endswith(".npy"):
                npy_files.append((file, target_speaker_id))
        if npy_files:
            with open(os.path.join(subdir_path, "_frames.txt"), "w") as f:
                for frames, target in npy_files:
                    f.write(frames + ", " + target + "\n")
    print("'_frames.txt' files created at '%s' directory " % dir_path)
