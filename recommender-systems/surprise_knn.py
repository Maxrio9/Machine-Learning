import pandas as pd
from surprise import Reader

book_ratings = pd.read_csv('goodreads_ratings.csv')
print(book_ratings.head())

#1. Print dataset size and examine column data types
print(book_ratings.info())
#2. Distribution of ratings
print(book_ratings.rating.value_counts())
#3. Filter ratings that are out of range
book_ratings = book_ratings[book_ratings.rating != 0]
#4. Prepare data for surprise: build a Suprise reader object
from surprise import Reader
reader = Reader(rating_scale = (1,5))
#5. Load `book_ratings` into a Surprise Dataset
from surprise import Dataset
data = Dataset.load_from_df(book_ratings[["user_id", "book_id", "rating"]], reader)
#6. Create a 80:20 train-test split and set the random state to 7
from surprise.model_selection import train_test_split
trainset, testset = train_test_split(data, test_size = 0.2, random_state = 7)
#7. Use KNNBasice from Surprise to train a collaborative filter
from surprise import KNNBasic
recommender = KNNBasic()
recommender.fit(trainset)

#8. Evaluate the recommender system
from surprise import accuracy
predictions = recommender.test(testset)
accuracy.rmse(predictions)

#9. Prediction on a user who gave the "The Three-Body Problem" a rating of 5
print(recommender.predict('8842281e1d1347389f2ab93d60773d4d', '18007564').est)

#10. Hyperparameter Tuning
from surprise.model_selection import GridSearchCV

param_grid = {
    'k': [20, 30, 40, 50],
    'sim_options': {'name': ['cosine', 'pearson'], 'user_based': [True, False]}
}

gs = GridSearchCV(KNNBasic, param_grid, measures=['rmse', 'mae'], cv=5)
gs.fit(data)

print(gs.best_score['rmse'])
print(gs.best_params['rmse'])