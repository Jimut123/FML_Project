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

print(genre_df.head(7))

user_df = pd.read_csv(r"ml-100k\ml-100k\u.user", sep="|",
                      names=["userID", "age", "gender", "occupation", "zip_code"])
movie_df = pd.read_csv(r"ml-100k\ml-100k\u.item", sep="|", names=["itemID", "title", "release_date", "video_release_date", "IMDb_URL", "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"],encoding='latin-1')
print(user_df.head(7))


def user_profile(type_wise):
    if (type_wise[0] == "Gender Statistics"):
        #px.pie(df, values='pop', names='country')
        fig = px.pie(user_df, names="gender", values='userID')
        fig.update_layout(
            title="User Profile Distribution based on Gender",
            xaxis_title="Gender",
            yaxis_title="Count",
        )
        return fig
    elif (type_wise[0] == "Age Statistics"):
        fig = px.bar(user_df, "age", 'userID')
        fig.update_layout(
            title="User Profile Distribution based on Age",
            xaxis_title="Age",
            yaxis_title="Count",
        )
        return fig
    elif (type_wise[0] == "Occupation Statistics"):
        fig = px.pie(user_df, names="occupation", values='userID')
        fig.update_layout(
            title="User Profile Distribution based on Occupation",
            xaxis_title="Occupation",
            yaxis_title="Count",
        )
        return fig
    else:
        raise ValueError("A plot type must be selected")


def outbreak(plot_type):

   
    if plot_type[0] == "Genre Distribution":
        fig = px.bar(genre_df, x="genreName", y='count')
        fig.update_layout(
            title="Genre Distribution",
            xaxis_title="Genre",
            yaxis_title="Count",
        )
        return fig
    elif(plot_type[0] == "Movie Distribution"):
        fig = px.pie(movie_df, "release_date", 'itemID')
        fig.update_layout(
            title="Movie Distribution",
            xaxis_title="Movie",
            yaxis_title="Count",
        )
        return fig
        
    else:
        raise ValueError("A plot type must be selected")


inputs = [
    # gr.Dropdown(["Matplotlib", "Plotly"], label="Plot Type"),
    gr.CheckboxGroup(["Genre Distribution","Movie Distribution"], label="Plot Type", value=[
                     "Genre Distribution"]),

]
inputs_user = [

    gr.CheckboxGroup(["Age Statistics", "Gender Statistics", "Occupation Statistics"],
                     label="Type Wise", value=["Occupation Statistics"]),

]
outputs_user = gr.Plot()
outputs = gr.Plot()

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
    ### User Profile Statistics
    #### Select a user to get recommendations for.
    """)

        gr.Interface(
            fn=user_profile,
            inputs=inputs_user,
            outputs=outputs_user,
            cache_examples=True,
        )

    with gr.Box():
        gr.Markdown(
            """
    ### Movies Statistics
    #### Select a statistics type to get the distribution.
    """)

        gr.Interface(
            fn=outbreak,
            inputs=inputs,
            outputs=outputs,
            cache_examples=True,
        )

if __name__ == "__main__":
    demo.launch()
