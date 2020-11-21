# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
from contextlib import contextmanager
from tempfile import TemporaryDirectory
import requests
import math
import shutil
import zipfile
import tarfile
from tqdm import tqdm
import logging

log = logging.getLogger(__name__)

def maybe_download(url, filename=None, work_directory="."):
    """Download a file if it is not already downloaded.
    Args:
        filename (str): File name.
        work_directory (str): Working directory.
        url (str): URL of the file to download.
        expected_bytes (int): Expected file size in bytes.

    Returns:
        str: File path of the file downloaded.
    """
    if filename is None:
        filename = url.split("/")[-1]
    os.makedirs(work_directory, exist_ok=True)
    filepath = os.path.join(work_directory, filename)
    if not os.path.exists(filepath):
        r = requests.get(url, stream=True)
        total_size = int(r.headers.get("content-length", 0))
        block_size = 1024
        num_iterables = math.ceil(total_size / block_size)
        with open(filepath, "wb") as file:
            for data in tqdm(
                    r.iter_content(block_size),
                    total=num_iterables,
                    unit="KB",
                    unit_scale=True,
            ):
                file.write(data)
    else:
        log.info("File {} already downloaded".format(filepath))
    print('file dir : ', filepath)
    return filepath


@contextmanager
def download_path(path_to_cache=None):
    """Return a path to download data. If `path=None`, then it yields a temporal path that is eventually deleted,
    otherwise the real path of the input.
    Args:
        path_to_cache (str): Path to download data.
    Returns:
        str: Real path where the data is stored.
    """
    if path_to_cache is None:
        tmp_dir = TemporaryDirectory()
        try:
            yield tmp_dir.name
        finally:
            tmp_dir.cleanup()
    else:
        path_to_cache = os.path.realpath(path_to_cache)
        yield path_to_cache


def unzip_file(zip_src, dst_dir, clean_zip_file=True):
    """Unzip a file
    Args:
        zip_src (str): Zip file.
        dst_dir (str): Destination folder.
        clean_zip_file (bool): Whether or not to clean the zip file.
    """
    fz = zipfile.ZipFile(zip_src, "r")
    for file in fz.namelist():
        fz.extract(file, dst_dir)
    if clean_zip_file:
        os.remove(zip_src)


def download_data(url, dest_path):
    dirs, file = os.path.split(dest_path)
    filepath = maybe_download(url, file, work_directory=dirs)
    return filepath


def extract_file_from_zip(zip_path, file_path, path = None):
    _, file_name = os.path.split(file_path)
    dirs, _ = os.path.split(zip_path)
    if path is None:
        dest_path = os.path.join(dirs, file_name)
    else:
        dest_path = os.path.join(path, file_name)

    with zipfile.ZipFile(zip_path, "r") as z:
        with z.open(file_path) as zf, open(dest_path, "wb") as f:
            shutil.copyfileobj(zf, f)

    return dest_path


def extract_file_from_tar(tar_path, file_name, path = None):
    if path is None:
        folder = os.path.dirname(tar_path)
        dest_path = os.path.join(folder, "dac")
    else:
        dest_path = path

    with tarfile.open(tar_path) as tar:
        tar.extractall(dest_path)
    return os.path.join(dest_path, file_name)

