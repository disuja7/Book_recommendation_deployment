from flask import Flask, render_template, request,app,jsonify,url_for,redirect,session,escape
import numpy as np
import scipy
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
import pickle
import nltk

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

# Defining the title_search function
import nltk
import string
from nltk.corpus import stopwords
translator = str.maketrans('', '', string.punctuation)
nltk.download('stopwords')
sw = stopwords.words('english')
def recommend(book_name):
    book_name =  book_name.lower().strip()
    book_name = book_name.replace(',',' ')
    book_name = book_name.replace(':',' ')
    name = []
    match_count = []
    for word in book_name.split():
        if word.lower() not in sw:
            name.append(word.lower() )
    title_list = []
    for elem in name:
      title = book_search_df[book_search_df['Book-Title'].str.contains(elem)]['Book-Title'].values
      if len(title)>0:
        title_list.extend(set(list(title)))   
    for title in title_list:
        title=title.translate(translator)
        match_name = []
        for word in title.split():
            if word.lower() not in sw:
                match_name.append(word.lower() )
        match_count.append(len(set(name)&set(match_name)))
    try:
        index = match_count.index(max(match_count))
        title = title_list[index]
    except:
        title = "No title found"
    try:
        #serach index
        index = np.where(users_items_pivot_matrix_df.index==title)[0][0]
        similar_items = sorted(list(enumerate(similarity_scores[index])),key=lambda x:x[1],reverse=True)[1:6]
        actual_data = []
        recommended_data = []
        for i in similar_items:
            item = []
            temp_df = book_search_df[book_search_df['Book-Title'] == users_items_pivot_matrix_df.index[i[0]]]
            item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Title'].values))
            item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Author'].values))
            item.extend(list(temp_df.drop_duplicates('Book-Title')['Image-URL-M'].values))
            recommended_data.append(item)
        actual_item = []
        actual_df = book_search_df[book_search_df['Book-Title'] == title]
        actual_item.extend(list(actual_df.drop_duplicates('Book-Title')['Book-Title'].values))
        actual_item.extend(list(actual_df.drop_duplicates('Book-Title')['Book-Author'].values))
        actual_item.extend(list(actual_df.drop_duplicates('Book-Title')['Image-URL-M'].values))
        actual_data.append(actual_item)

        final_data = []
        final_data.append(actual_data)
        final_data.append(recommended_data)

        return final_data

    except Exception as error:
        return []



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
    print(recommendations)
    return render_template('recommend.html', data = recommendations)




if __name__ == '__main__':
    app.run(debug = True)
