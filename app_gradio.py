from matplotlib import cm
import pandas as pd
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
import gradio as gr
from math import sqrt
import matplotlib
matplotlib.style.use('ggplot')

matplotlib.use("Agg")


genre_df = pd.read_csv(r"ml-100k\ml-100k\u.genre",
                       sep="|", names=["genreName", "count"])

ratings_df = pd.read_csv(r"ml-100k\ml-100k\u.data", sep="\t",
                         names=["userID", "itemID", "rating", "timestamp"])

sorted_ratingsdf = ratings_df.sort_values(by=['rating'], ascending=False)

user_df = pd.read_csv(r"ml-100k\ml-100k\u.user", sep="|",
                      names=["userID", "age", "gender", "occupation", "zip_code"])
movie_df = pd.read_csv(r"ml-100k\ml-100k\u.item", sep="|", names=["itemID", "title", "release_date", "video_release_date", "IMDb_URL", "unknown", "Action", "Adventure", "Animation",
                       "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"], encoding='latin-1')



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


def movieStatistics(plot_type):

    if plot_type[0] == "Genre Distribution":
        fig = px.bar(genre_df, x="genreName", y='count')
        fig.update_layout(
            title="Genre Distribution",
            xaxis_title="Genre",
            yaxis_title="Count",
        )
        return fig
    elif (plot_type[0] == "Movie Distribution"):
        movie_analysis_df = movie_df
        movie_analysis_df['yearOfRelease'] = movie_df.title.apply(
        lambda x: int(x[-5:-1]) if x[-5:-1].isdigit() else 1990)
        df_year_genre = movie_analysis_df.groupby('yearOfRelease')["unknown", "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
                                                            "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"].sum()
        df_year_genre = df_year_genre.reset_index()

        print(df_year_genre.head())
        x1 = df_year_genre[-50:-1]  # considering data for last 50 years

        #my_colors = [(x/10.0, x/20.0, 0.75) for x in range(1,19)]
        cmap = cm.get_cmap('Spectral')
        ax = px.bar(x1,x='yearOfRelease', y=["Action", "Adventure", "Animation", "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
                                        ], title='movies produced every year for each genre')
        plt.legend(loc=10, bbox_to_anchor=(1.2, .5), ncol=1)
        #ax.set_xlabel('Year')
        #ax.set_ylabel('Count of Movies')
        ax.update_layout(
        xaxis_title="Year",
        yaxis_title="Count of Movies",
        )
        ax.write_image('movie_genre.svg', format='svg')
        return ax

    else:
        raise ValueError("A plot type must be selected")


def userBehavior(plot_type):

     
    
    movie_analysis_df = movie_df
    movie_analysis_df['yearOfRelease'] = movie_df.title.apply(
        lambda x: int(x[-5:-1]) if x[-5:-1].isdigit() else 1990)
    df_year_genre = movie_analysis_df.groupby('yearOfRelease')["unknown", "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
                                                            "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"].sum()
    df_year_genre = df_year_genre.reset_index()

    print(df_year_genre.head())
    x1 = df_year_genre[-50:-1]  # considering data for last 50 years

    #my_colors = [(x/10.0, x/20.0, 0.75) for x in range(1,19)]
    cmap = cm.get_cmap('Spectral')
    ax = px.bar(x1,x='yearOfRelease', y=["Action", "Adventure", "Animation", "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
                                        ], title='movies produced every year for each genre')
    plt.legend(loc=10, bbox_to_anchor=(1.2, .5), ncol=1)
    #ax.set_xlabel('Year')
    #ax.set_ylabel('Count of Movies')
    ax.update_layout(
    xaxis_title="Year",
    yaxis_title="Count of Movies",
    )
    ax.write_image('movie_genre.svg', format='svg')
    return ax


inputs = [
    # gr.Dropdown(["Matplotlib", "Plotly"], label="Plot Type"),
    gr.CheckboxGroup(["Genre Distribution", "Movie Distribution"], label="Plot Type", value=[
                     "Genre Distribution"]),

]
inputs_user = [

    gr.CheckboxGroup(["Age Statistics", "Gender Statistics", "Occupation Statistics"],
                     label="Type Wise", value=["Occupation Statistics"]),

]

inputs_user_behavior = [
    
    gr.CheckboxGroup(["movie genre"], label="Plot Type", value=["movie genre"])]
  
outputs_movie = gr.Plot()
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
            fn=movieStatistics,
            inputs=inputs,
            outputs=outputs,
            cache_examples=True,
        )
        


if __name__ == "__main__":
    demo.launch()
