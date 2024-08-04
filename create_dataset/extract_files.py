import os
from shutil import copy2
from tqdm import tqdm


def copy_files(src_artist, dest_artist):
    """
    Copy files from subdirectories into new subdirectory for each artist
    :param src_artist: Source artist folder with subdirectories
    :arg src_artist: str
    :param dest_artist: Destination folder where new subdirectories will be created
    :arg dest_artist: str
    :return: None
    """
    
    artist_src_dir = os.path.join(root_dir, src_artist)
    artist_destination_dir = os.path.join(destination_dir, dest_artist)
    
    os.makedirs(artist_destination_dir, exist_ok=True)  # Create a new subdirectory for the artist
    
    print(f"Copying files for {src_artist}")
    for root, dirs, files in os.walk(artist_src_dir):
        for file in tqdm(files):
            path_file = os.path.join(root, file)
            copy2(path_file, artist_destination_dir)
            
    print("Done")


if __name__ == "__main__":

    src_artist_lst = ['claude-lorrain', 'claude-monet', 'edgar-degas', 'edvard-munch', 
                      'nicolas-poussin', 'pierre-auguste-renoir', 'vincent-van-gogh']

    dest_artist_lst = ['Lorrain', 'Monet', 'Degas', 'Munch',
                       'Poussin', 'Renoir', 'VanGogh']

    # Directories specific to my local machine
    root_dir = '/Users/Administrator/Desktop/wikiart/wikiart-saved/images/'
    destination_dir = '/Users/Administrator/Desktop/thesis/'

    zip_lst = list(zip(src_artist_lst, dest_artist_lst))

    for (src, dest) in zip_lst:
        copy_files(src, dest)
