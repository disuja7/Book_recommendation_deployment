# Importing necessary modules
import numpy as np
import scipy
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
import pickle


# import os

# if os.path.getsize(target) > 0:      
#     with open(target, "rb") as f:
#         unpickler = pickle.Unpickler(f)
#         # if file is not empty scores will be equal
#         # to the value unpickled
#         scores = unpickler.load()


pred_df = pickle.load(open('pred_df.pkl','rb'))
books = pickle.load(open('books.pkl','rb'))

# preprocess_and_model = pickle.load(open('preprocess_and_model.pkl','rb'))

# # fetch data after preprocessing and applying SVD as pred_df
# pred_df = preprocess_and_model(df)

recommend_items_by_item = pickle.load(open('recommend_items_by_item.pkl','rb'))
recommend = pickle.load(open('recommend.pkl','rb'))

recommended_books = recommend('shiv')

print(preprocess_and_model)

