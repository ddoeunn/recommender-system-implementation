import os
from collections import namedtuple
import pandas as pd
from utils.dataset.download_utils import (
    download_path, download_data, extract_file_from_zip
)
from utils.common.constants import (
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
    DEFAULT_HEADER
)

URL_MOVIE_LENS = "http://files.grouplens.org/datasets/movielens/"
ERROR_MOVIE_LENS_SIZE = "Invalid data size. Should be one of {'100k', '1m', '10m', '20m'}"
ERROR_HEADER = """Invalid data header. It consists of four columns (user_id, movie_id, rating, timestamp)
    and should contain at least user_id and movie_id."""


MovieLens = namedtuple("MovieLens", ["rating_path", "item_path",
                                     "rating_sep", "item_sep",
                                     "rating_header", "item_header"])
ML_FORMAT = {
    "100k": MovieLens(
        "ml-100k/u.data", "ml-100k/u.item", "\t", "|", False, False
    ),
    "1m": MovieLens(
        "ml-1m/ratings.dat", "ml-1m/movies.dat", "::", "::", False, False
    ),
    "10m": MovieLens(
        "ml-10M100K/ratings.dat", "ml-10M100K/movies.dat", "::", "::", False, False
    ),
    "20m": MovieLens(
        "ml-20m/ratings.csv", "ml-20m/movies.csv", ",", ",", True, True
    ),
}

# 100K data genres index to string mapper. For 1m, 10m, and 20m, the genres labels are already in the dataset.
GENRES = (
    "unknown",
    "Action",
    "Adventure",
    "Animation",
    "Children's",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
)

def load_data(size="100k",
    header=None,
    local_cache_path=None,
    unzip_path=None,
    title_col=None,
    genres_col=None,
    year_col=None,):
    """

    :param size: One of ("100k", "1m", "10m", "20m"). size of data to load
    :param header:
    :param path_to_cache:
    :param title_col:
    :param genres_col:
    :param year_col:
    :return:
    """
    size = size.lower()
    if size not in ML_FORMAT:
        raise ValueError(ERROR_MOVIE_LENS_SIZE)

    if header is None:
        header = DEFAULT_HEADER
    elif (len(header) < 2) or (len(header) > 4):
        raise ValueError(ERROR_HEADER)

    url = URL_MOVIE_LENS + 'ml-' + size + ".zip"
    with download_path(local_cache_path) as path:
        zip_path = os.path.join(path, "ml-{}.zip".format(size))
        filepath = download_data(url, zip_path)
        rating_path = extract_file_from_zip(filepath, ML_FORMAT[size].rating_path, unzip_path)
        item_path = extract_file_from_zip(filepath, ML_FORMAT[size].item_path, unzip_path)

        # Load rating data
        rating_df = pd.read_csv(
            rating_path,
            sep=ML_FORMAT[size].rating_sep,
            engine="python",
            names=header,
            usecols=[*range(len(header))],
            header=0 if ML_FORMAT[size].rating_header else None,
        )

        # Load movie features such as title, genres, and release year
        item_df = load_item_df(
            size, item_path, DEFAULT_ITEM_COL, title_col, genres_col, year_col
        )

        # Convert 'rating' type to float
        if len(header) > 2:
            rating_df[header[2]] = rating_df[DEFAULT_USER_COL].astype(float)

        # Merge rating df w/ item_df
        if item_df is not None:
            df = rating_df.merge(item_df, on=DEFAULT_ITEM_COL)
            return df
        else:
            return rating_df

def load_item_df(size, item_datapath, movie_col, title_col, genres_col, year_col):
    """Loads Movie info"""
    if title_col is None and genres_col is None and year_col is None:
        return None

    item_header = [movie_col]
    usecols = [0]

    # Year is parsed from title
    if title_col is not None or year_col is not None:
        item_header.append("title_year")
        usecols.append(1)

    genres_header_100k = None
    if genres_col is not None:
        # 100k data's movie genres are encoded as a binary array (the last 19 fields)
        # For details, see http://files.grouplens.org/datasets/movielens/ml-100k-README.txt
        if size == "100k":
            genres_header_100k = [*(str(i) for i in range(19))]
            item_header.extend(genres_header_100k)
            usecols.extend([*range(5, 24)])  # genres columns
        else:
            item_header.append(genres_col)
            usecols.append(2)  # genres column

    item_df = pd.read_csv(
        item_datapath,
        sep=ML_FORMAT[size].item_sep,
        engine="python",
        names=item_header,
        usecols=usecols,
        header=0 if ML_FORMAT[size].item_header else None,
        encoding="ISO-8859-1",
    )

    # Convert 100k data's format: '0|0|1|...' to 'Action|Romance|..."
    if genres_header_100k is not None:
        item_df[genres_col] = item_df[genres_header_100k].values.tolist()
        item_df[genres_col] = item_df[genres_col].map(
            lambda l: "|".join([GENRES[i] for i, v in enumerate(l) if v == 1])
        )

        item_df.drop(genres_header_100k, axis=1, inplace=True)

    # Parse year from movie title. Note, MovieLens title format is "title (year)"
    # Note, there are very few records that are missing the year info.
    if year_col is not None:

        def parse_year(t):
            parsed = re.split("[()]", t)
            if len(parsed) > 2 and parsed[-2].isdecimal():
                return parsed[-2]
            else:
                return None

        item_df[year_col] = item_df["title_year"].map(parse_year)
        if title_col is None:
            item_df.drop("title_year", axis=1, inplace=True)

    if title_col is not None:
        item_df.rename(columns={"title_year": title_col}, inplace=True)

    return item_df


