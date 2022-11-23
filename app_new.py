import pandas as pd
import numpy as np
from zipfile import ZipFile
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
import matplotlib.pyplot as plt
import gradio as gr


## Dataset based visualizations

genre_df = pd.read_csv(r"ml-100k\ml-100k\u.genre", sep="|", names=["genreName", "count"])    

print(genre_df.head(7))

user_df = pd.read_csv(r"ml-100k\ml-100k\u.user", sep="\t", names=["userID", "age", "gender", "occupation" ])    
#movie_df = pd.read_csv(r"ml-100k\ml-100k\u.item", sep="\t", names=["itemID", "title", "release_date", "video_release_date", "IMDb_URL", "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"])     
   


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

    #inp1 = gr.Slider(0, num_users-1, value=0, label='User')
    # btn1 = gr.Button('Random User')
    gr.Plot(x=genre_df['genreName'], y=genre_df['count'], title='Genre Distribution', x_label='Genre', y_label='Count')
    # top_rated_from_user = get_top_rated_from_user(0)
   
    #df1 = gr.DataFrame(headers=["title", "genres"], datatype=["str", "str"], interactive=False)

  

demo.launch(debug=True)