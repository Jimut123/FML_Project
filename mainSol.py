data_sizes = ["100k"] # Movielens data size: 100k, 1m, 10m, or 20m
for data_size in data_sizes:
    # Load the dataset
    df = movielens.load_pandas_df(
        size=data_size,
        header=[DEFAULT_USER_COL, DEFAULT_ITEM_COL, DEFAULT_RATING_COL, DEFAULT_TIMESTAMP_COL]
    )
    print("Size of Movielens {}: {}".format(data_size, df.shape))
    