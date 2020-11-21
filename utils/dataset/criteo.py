import os
from collections import namedtuple
import pandas as pd

from utils.dataset.download_utils import (
    download_path, download_data, extract_file_from_tar
)

URL_CTR_FULL = "https://s3-eu-west-1.amazonaws.com/kaggle-display-advertising-challenge-dataset/dac.tar.gz"
URL_CTR_SAMPLE = "http://labs.criteo.com/wp-content/uploads/2015/04/dac_sample.tar.gz"
CTR_HEADER = (
    ["label"]
    + ["int{0:02d}".format(i) for i in range(13)]
    + ["cat{0:02d}".format(i) for i in range(26)]
)

CTR = namedtuple("CTR", ["url", "tar_file", "data_file"])
CTR_FORMAT = {
    "full": CTR(
        URL_CTR_FULL, "dac.tar.gz", "train.txt"
    ),
    "sample": CTR(
        URL_CTR_SAMPLE, "dac_sample.tar.gz", "dac_sample.txt"
    )
}

def load_data(size="sample", local_cache_path=None, unzip_path=None, header=CTR_HEADER):
    size = size.lower()
    url = CTR_FORMAT[size].url
    tar_file = CTR_FORMAT[size].tar_file
    data_file = CTR_FORMAT[size].data_file
    with download_path(local_cache_path) as path:
        tar_path = os.path.join(path, tar_file)
        filepath = download_data(url, tar_path)
        filepath = extract_file_from_tar(filepath, data_file, unzip_path)
        df = pd.read_csv(filepath, sep="\t", header=None, names=header)

        return df
