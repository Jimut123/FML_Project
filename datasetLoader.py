import os
import pandas as pdf

from utils.downloaderData import download_path



def _maybe_download_and_extract(size, dest_path):
    """Downloads and extracts MovieLens rating and item datafiles if they don’t already exist"""
    dirs, _ = os.path.split(dest_path)
    if not os.path.exists(dirs):
        os.makedirs(dirs)

    _, rating_filename = os.path.split(DATA_FORMAT[size].path)
    rating_path = os.path.join(dirs, rating_filename)
    _, item_filename = os.path.split(DATA_FORMAT[size].item_path)
    item_path = os.path.join(dirs, item_filename)

    if not os.path.exists(rating_path) or not os.path.exists(item_path):
        download_movielens(size, dest_path)
        extract_movielens(size, rating_path, item_path, dest_path)

    return rating_path, item_path


def download_movielens(size, dest_path):
    """Downloads MovieLens datafile.
    Args:
        size (str): Size of the data to load. One of ("100k", "1m", "10m", "20m").
        dest_path (str): File path for the downloaded file
    """
    if size not in DATA_FORMAT:
        raise ValueError(ERROR_MOVIE_LENS_SIZE)

    url = "https://files.grouplens.org/datasets/movielens/ml-" + size + ".zip"
    dirs, file = os.path.split(dest_path)
    maybe_download(url, file, work_directory=dirs)


def extract_movielens(size, rating_path, item_path, zip_path):
    """Extract MovieLens rating and item datafiles from the MovieLens raw zip file.
    To extract all files instead of just rating and item datafiles,
    use ZipFile's extractall(path) instead.
    Args:
        size (str): Size of the data to load. One of ("100k", "1m", "10m", "20m").
        rating_path (str): Destination path for rating datafile
        item_path (str): Destination path for item datafile
        zip_path (str): zipfile path
    """
    with ZipFile(zip_path, "r") as z:
        with z.open(DATA_FORMAT[size].path) as zf, open(rating_path, "wb") as f:
            shutil.copyfileobj(zf, f)
        with z.open(DATA_FORMAT[size].item_path) as zf, open(item_path, "wb") as f:
            shutil.copyfileobj(zf, f)


def load_pandas_df(
    size="100k",
    header=None,
    local_cache_path=None,
    title_col=None,
    genres_col=None,
    year_col=None,
):
    """Loads the MovieLens dataset as pd.DataFrame.
    Download the dataset from https://files.grouplens.org/datasets/movielens, unzip, and load.
    To load movie information only, you can use load_item_df function.
    Args:
        size (str): Size of the data to load. One of ("100k", "1m", "10m", "20m", "mock100").
        header (list or tuple or None): Rating dataset header.
            If `size` is set to any of 'MOCK_DATA_FORMAT', this parameter is ignored and data is rendered using the 'DEFAULT_HEADER' instead.
        local_cache_path (str): Path (directory or a zip file) to cache the downloaded zip file.
            If None, all the intermediate files will be stored in a temporary directory and removed after use.
            If `size` is set to any of 'MOCK_DATA_FORMAT', this parameter is ignored.
        title_col (str): Movie title column name. If None, the column will not be loaded.
        genres_col (str): Genres column name. Genres are '|' separated string.
            If None, the column will not be loaded.
        year_col (str): Movie release year column name. If None, the column will not be loaded.
            If `size` is set to any of 'MOCK_DATA_FORMAT', this parameter is ignored.
    Returns:
        pandas.DataFrame: Movie rating dataset.
    **Examples**
    .. code-block:: python
        # To load just user-id, item-id, and ratings from MovieLens-1M dataset,
        df = load_pandas_df('1m', ('UserId', 'ItemId', 'Rating'))
        # To load rating's timestamp together,
        df = load_pandas_df('1m', ('UserId', 'ItemId', 'Rating', 'Timestamp'))
        # To load movie's title, genres, and released year info along with the ratings data,
        df = load_pandas_df('1m', ('UserId', 'ItemId', 'Rating', 'Timestamp'),
            title_col='Title',
            genres_col='Genres',
            year_col='Year'
        )
    """
    size = size.lower()
    
    if header is None:
        header = DEFAULT_HEADER
    elif len(header) < 2:
        raise ValueError(ERROR_HEADER)
    elif len(header) > 4:
        warnings.warn(WARNING_MOVIE_LENS_HEADER)
        header = header[:4]


    movie_col = header[1]

    with download_path(local_cache_path) as path:
        filepath = os.path.join(path, "ml-{}.zip".format(size))
        datapath, item_datapath = _maybe_download_and_extract(size, filepath)

        # Load movie features such as title, genres, and release year
        item_df = _load_item_df(
            size, item_datapath, movie_col, title_col, genres_col, year_col
        )

        # Load rating data
        df = pd.read_csv(
            datapath,
            sep=DATA_FORMAT[size].separator,
            engine="python",
            names=header,
            usecols=[*range(len(header))],
            header=0 if DATA_FORMAT[size].has_header else None,
        )

        # Convert 'rating' type to float
        if len(header) > 2:
            df[header[2]] = df[header[2]].astype(float)

        # Merge rating df w/ item_df
        if item_df is not None:
            df = df.merge(item_df, on=header[1])

    return df