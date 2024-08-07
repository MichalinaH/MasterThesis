"""
Script to create train and valid splits for each individual artist
"""

import os
import random
from shutil import copy2
from tqdm import tqdm


def split_data(root, artist_name, train_split_idx, valid_split_idx):
    """
    Split data into train and valid sets
    :param root: Root directory
    :type root: str
    :param artist_name: Artist name
    :type artist_name: str
    :param train_split_idx: Index of train file split
    :type train_split_idx: int
    :param valid_split_idx: Index of valid file split
    :type valid_split_idx: int
    :return: None
    """


    print(f"Creating train and validation splits for {artist_name}")

    src = os.path.join(root, artist_name)
    train_dir = os.path.join(root, 'training', artist_name)
    valid_dir = os.path.join(root, 'validation', artist_name)

    # Create artist-specific train and validation directories
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(valid_dir, exist_ok=True)


    # Check for zero length files
    files = []
    for file in os.listdir(src):
        if os.path.getsize(f"{src}/{file}") > 0:
            files.append(file)
        else:
            print(f"{file} is zero length, so ignoring.")

    # Shuffle files
    files_shuffled = random.sample(files, len(files))

    for file in tqdm(files_shuffled[:train_split_idx]):
        path_file = os.path.join(src, file)
        copy2(path_file, train_dir)

    for file in tqdm(files_shuffled[-valid_split_idx:]):
        path_file = os.path.join(src, file)
        copy2(path_file, valid_dir)


def check_files(root, artist_name):
    """
    Function to verify that train and valid files are distinct and to create directories of
    non train, valid files for each artist for post modeling verification
    :param root: Root directory
    :arg root: str
    :param artist_name: Artist name
    :arg artist_name: str
    :return: None
    """

    print(f"Checking for duplicates for {artist_name}")

    src_dir = os.path.join(root, artist_name)
    test_dir = os.path.join(root, 'files_for_testing', artist_name)

    os.makedirs(test_dir, exist_ok=True)
    
    train_dir = os.path.join(root, 'training', artist_name)
    valid_dir = os.path.join(root, 'validation', artist_name)

    all_files = os.listdir(src_dir)
    train_files = os.listdir(train_dir)
    valid_files = os.listdir(valid_dir)

    duplicate_files = len(set(train_files) & set(valid_files))

    print(f"{duplicate_files} duplicates for {artist_name}")
    
    not_train_valid_lst = []
    
    for file in all_files:
        if (file not in train_files) and (file not in valid_files):
            not_train_valid_lst.append(file)
            
    for file in not_train_valid_lst:
        path_file = os.path.join(src_dir, file)
        copy2(path_file, test_dir)
        
    print(f"Finished creating non train and non valid samples for {artist_name}")       
            

if __name__ == "__main__":

    # Root directory specific to my local machine
    root_dir = '/Users/Administrator/Desktop/thesis'

    artist_lst = ['Lorrain', 'Monet', 'Degas', 'Munch',
                'Poussin', 'Renoir', 'VanGogh']


    train_ratio = 0.75
    valid_ratio = 0.2

    for artist in artist_lst:

        src_dir = os.path.join(root_dir, artist)
        num_files = len(os.listdir(src_dir))
        
        train_idx = int(train_ratio * num_files)
        valid_idx = int(valid_ratio * num_files)
        
        split_data(root_dir, artist, train_idx, valid_idx)

    for artist in artist_lst:
        check_files(root_dir, artist)
