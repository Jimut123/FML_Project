from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm
from LightGCN import LightGCN
from dataModel import ImplicitCF


df = pd.read_csv(r"ml-100k\ml-100k\u.data", sep="\t", names=["userID", "itemID", "rating", "timestamp"])    
    

rating_df = df[df["rating"]>=1.0]
    # Split the dataset

df_train, df_test = train_test_split(rating_df, test_size
                               =0.25)


def prepare_training_lightgcn(train, test):
    return ImplicitCF(train=train, test=test)


params ={'n_layers': 3,
 'epochs': 15,
 'learning_rate': 0.005,
 'eval_epoch': 5,
 'top_k': 10,
 'embed_size' : 64,
 'metrics' : ["recall", "ndcg", "precision", "map"], # metrics for evaluation
  'batch_size' : 1024, # batch size for training
    'decay' : 0.0001 # l2 regularization for embedding parameters
 } 

def train_lightgcn(params, data):
    hparams = params
    # defaultMode = true (Orignial LightGCN model)
    # defaultMode = false (LightGCN++ model)
    model = LightGCN(hparams, data, defaultMode= False)
    with tqdm() as t:
        model.fit()
    return model, t

train = prepare_training_lightgcn(df_train, df_test)

model, time_train = train_lightgcn(params, train)

def recommend_k_lightgcn(model, test, top_k=10, remove_seen=True):
    with tqdm() as t:
        topk_scores = model.recommendKItemsForallUsers(
            test, top_k=top_k, remove_seen=remove_seen
        )
    return topk_scores, t


top_k_scores, time_ranking = recommend_k_lightgcn(model, df_test)

print(top_k_scores.to_csv('pred.csv'))