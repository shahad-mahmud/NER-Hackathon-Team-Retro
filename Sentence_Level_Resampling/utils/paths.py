import os

def check_dir(dir_path: str):
    if os.path.exists(dir_path) and os.listdir(dir_path):
        raise ValueError(
            "Output directory ({}) already exists and is not empty.".format(dir_path))
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)