from flask import Flask, render_template, request,app,jsonify,url_for,redirect,session,escape
import numpy as np
import scipy
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
import pickle

app = Flask(__name__)

# getpredictions dataframe and books dataframe
book_search_df = pickle.load(open('book_search_df.pkl','rb'))
final_rating = pickle.load(open('final_rating.pkl','rb'))
popular_df = pickle.load(open('popular_df.pkl','rb'))

#Creating a sparse pivot table with users in rows and items in columns
users_items_pivot_matrix_df = final_rating.pivot_table(index='Book-Title', 
                                                          columns='User-ID', 
                                                          values='Book-Rating').fillna(0)
# getting similarity score
similarity_scores = cosine_similarity(users_items_pivot_matrix_df)

# Defining the recommendation function

def recommend(book_name):
    name1 =  book_name.lower().strip()
    name2 = name1.replace(',',' ')
    name3 = name2.replace(':',' ') 

    global title
    title = final_rating[final_rating['Book-Title'].str.contains(f'{name1}|{name2}|{name3}')]['Book-Title'].values
    try:
        title = title[0]
        #serach index
        index = np.where(users_items_pivot_matrix_df.index==title)[0][0]
        similar_items = sorted(list(enumerate(similarity_scores[index])),key=lambda x:x[1],reverse=True)[1:6]
        data = []
        for i in similar_items:
            item = []
            temp_df = book_search_df[book_search_df['Book-Title'] == users_items_pivot_matrix_df.index[i[0]]]
            item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Title'].values))
            item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Author'].values))
            item.extend(list(temp_df.drop_duplicates('Book-Title')['Image-URL-M'].values))

            data.append(item)
        return data
    except Exception as error:
        return f"Oops! Book Not Found.....\n Try Again\n{error}"


@app.route('/')
def home():
    return render_template('home.html',
                           book_name = list(popular_df['Book-Title'].values),
                           author = list(popular_df['Book-Author'].values),
                           image = list(popular_df['Image-URL-M'].values),
                           votes = list(popular_df['Rating_count'].values),
                           rating = list(popular_df['Avg_rating'].values)
                           )
                          

@app.route('/recommend')
def recommend_ui():
    return render_template('recommend.html')


@app.route('/recommend_api',methods=['post']) 
def recommend_api():
    user_input = request.form.get('Book_name')
    print(user_input)
    recommendations =  recommend(user_input)
    # recommendation_json = str(recommendation_df.to_json()).replace('\\','')
    # return render_template('recommend.html' , Book_name = Book, data = recommendation_json) 
    print(recommendations)
    return render_template('recommend.html',data=recommendations)

# @app.route('/recommend_static')
# def recommend_ui():




if __name__ == '__main__':
    app.run(debug = True)
