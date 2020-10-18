import os


def get_filesnames_from_directory(path, ext=None, full_path=False):
    files = [name for name in os.listdir(path)]
    if ext:
        files = [name for name in files if name.endswith(ext)]
    if full_path:
        files = [f"{path}/{name}" for name in files]
    return files
