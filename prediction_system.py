import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances

# Load Movies and their ratings dataset,Ensure that dataset should be copyright 
movies = pd.read_csv("movies.csv")# add your movies file path
ratings = pd.read_csv("ratings.csv")# add your ratings file path

movie_ratings = pd.merge(ratings, movies, on='movieId') # merge the both above dataset

# Creating a  user-item matrix(row for user and column for movies ratings)
user_movie_ratings = movie_ratings.pivot_table(index='userId', columns='title', values='rating')

user_movie_ratings = user_movie_ratings.fillna(0)# fill nil area with 0 (assume thar their is no ratings)

# Transpose user-item matrix for item-item collaborative filtering
movie_similarity = cosine_similarity(user_movie_ratings.T)

# Convert similarity matrix into a DataFrame
movie_similarity_df = pd.DataFrame(movie_similarity, index=user_movie_ratings.columns, columns=user_movie_ratings.columns)

# Main function to get movie recommendations for a given movie
def get_movie_recommendations(movie_title, user_ratings):
    similar_scores = movie_similarity_df[movie_title] * (user_ratings - 2.5)
    similar_movies = similar_scores.sort_values(ascending=False)
    return similar_movies


# Main function to generate movie recommendations for a user
def generate_recommendations(user_ratings):
    recommendations = pd.Series()
    for movie, rating in user_ratings.items():
        similar_scores = movie_similarity_df[movie] * (rating - 2.5)
        recommendations = recommendations.add(similar_scores, fill_value=0)

    return recommendations.sort_values(ascending=False)

# Example: Generate movie recommendations for a user
user_ratings = {'Inception (2010)': 5.0,'Dark Knight, The (2008)':4.0, 'Toy Story (1995)': 3.0}
recommendations = generate_recommendations(user_ratings)
print(recommendations.head(50))
