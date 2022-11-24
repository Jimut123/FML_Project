import pandas as pd
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
import gradio as gr
from math import sqrt
import matplotlib

matplotlib.use("Agg")


genre_df = pd.read_csv(r"ml-100k\ml-100k\u.genre",
                       sep="|", names=["genreName", "count"])


user_df = pd.read_csv(r"ml-100k\ml-100k\u.user", sep="|",
                      names=["userID", "age", "gender", "occupation", "zip_code"])
movie_df = pd.read_csv(r"ml-100k\ml-100k\u.item", sep="|", names=["itemID", "title", "release_date", "video_release_date", "IMDb_URL", "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"],encoding='latin-1')

prediction_df = pd.read_csv(r"pred.csv", sep=",")


def mappingMovie(mid):
    return movie_df.loc[movie_df["itemID"].values==mid]["title"].values[0]

prediction_df["Movie Title"] = prediction_df["itemID"].apply(mappingMovie)


df = pd.read_csv(r"ml-100k\ml-100k\u.data", sep="\t", names=["userID", "itemID", "rating", "timestamp"])    
    

rating_df = df[df["rating"]>=1.0]

rating_df["Movie Title"] = rating_df["itemID"].apply(mappingMovie)

print(rating_df.head(7))

num_users = len(prediction_df["userID"].unique())

def get_top_rated_movies_from_user(id):
    entire_df= rating_df[rating_df["userID"]==id].sort_values(by="rating", ascending=False).head(10)
    return entire_df
def get_recommendations(id):
    entire_df= prediction_df[prediction_df["userID"]==id].sort_values(by="prediction", ascending=False).head(10)
    entire_df.drop(columns=["Unnamed: 0"], inplace=True)
    return entire_df

def update_user(id):
    return get_top_rated_movies_from_user(id), get_recommendations(id)
def random_user():
    return update_user(np.random.randint(0, num_users-1))

demo = gr.Blocks()

with demo:
  gr.Markdown("""
  <div>
  <h1 style='text-align: center'>Movie Recommender</h1>
  Collaborative Filtering is used to predict the top 10 recommended movies for a particular user from the dataset based on that user and previous movies they have rated.
  
  Note: Currently there is a bug with sliders. If you "click and drag" on the slider it will not use the correct user. Please only "click" on the slider :D.
  </div>
  """)
    
  with gr.Box():
    gr.Markdown(
    """
    ### Input
    #### Select a user to get recommendations for.
    """)

    inp1 = gr.Slider(0, num_users-1, value=0, label='User')
    # btn1 = gr.Button('Random User')

    # top_rated_from_user = get_top_rated_from_user(0)
    gr.Markdown(
    """
    <br>
    """)
    gr.Markdown(
    """
    #### Movies with the Highest Ratings from this user
    """)
    df1 = gr.DataFrame(headers=["title", "genres"], datatype=["str", "str"], interactive=False)

  with gr.Box():
    # recommendations = get_recommendations(0)
    gr.Markdown(
    """
    ### Output
    #### Top 10 movie recommendations
    """)
    df2 = gr.DataFrame(headers=["title", "genres"], datatype=["str", "str"], interactive=False)

  gr.Markdown("""
  <p style='text-align: center'>
      <a href='https://keras.io/examples/structured_data/collaborative_filtering_movielens/' target='_blank' style='text-decoration: underline'></a>
      <br>
      Space by Scott Krstyen (mindwrapped)
      </p>
  """)
  
  
  inp1.change(fn=update_user,
              inputs=inp1,
              outputs=[df1, df2])
  

demo.launch(debug=True)