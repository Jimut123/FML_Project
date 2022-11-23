from sklearn.model_selection import train_test_split
import pandas as pd

from train_testSplit import python_stratified_split

df = pd.read_csv(r"ml-100k\ml-100k\u.data", sep="\t", names=["userID", "itemID", "rating", "timestamp"])    
    
print(df["rating"].head(20))

df = df[df["rating"]>=1.0]
    # Split the dataset

train, test = train_test_split(rating_df, test_size
                               =0.25)

print(df_train)