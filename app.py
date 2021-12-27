from flask import Flask
from flask import render_template
from flask import request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


#Reading filtered DataSet 
df = pd.read_csv('filtered_books.csv')

def gen_cos_matrix():
    try:
        cv = CountVectorizer(dtype=np.float32)
        book_matrix = cv.fit_transform(df['combined'])
        cos_similarity = cosine_similarity(book_matrix)
        np.save('cosine_matrix.npy', cos_similarity)
    except:
        print("There was some error while updating the matrix")

#gen_cos_matrix() #run once, comment this line later
cos_s = np.load('cosine_matrix.npy')

#Generating Cosine Scores and then selecting top 50 reccomnedations
def cos_scores(ind):
    similarity_scores = list(enumerate(cos_s[ind]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    similarity_scores = similarity_scores[1:51] 
    return similarity_scores

#From 50 recommendations we are filtering on the basis of average rating.
def filter_scores(cos_scores):
    averages = []
    for i, score in cos_scores:
        rating = df.loc[i, 'average_rating']
        average = (2 * score + (rating / 10)) / 3
        averages.append((i, average))
    averages = sorted(averages, key=lambda x: x[1], reverse=True)
    averages = averages[0:12]
    indexes = list(map(lambda item: item[0], averages))
    return indexes


indexes = pd.Series(df.index, index=df['title'])


#Getting the books from indexes
def books_names(df):
    books=[]
    for data in df['title']:
        books.append(data)
    return books
    
#Creating a dictonary type record for the recommended books.
def get_books_data(ind):
    book_data = []
    j = 1
    for i in ind:
        title = df.loc[i, 'title']
        year = df.loc[i, 'published_year']
        author= df.loc[i, 'authors']
        category=df.loc[i,"categories"]
        thumbnail=df.loc[i,'thumbnail']
        description =df.loc[i,'description']
        published_year=df.loc[i,'published_year']
        average_rating=df.loc[i,'average_rating']
        book_data.append({'id': j, 'title': title,'category':category, 'year': year, 'Author': author,
        'thumbnail' : thumbnail,'description':description, 'published_year': published_year, 'average_rating':average_rating})
        j = j + 1
    return book_data

#Integrating all the methods to create recommendations simultaneously.
def recommendations(title):
    rec_books = []
    if title in df['title'].unique():
        index = indexes[title]
        records = cos_scores(index)
        records = filter_scores(records)
        records = get_books_data(records)
        rec_books.append(records)
        return rec_books


#Chart Funtions :

# Barplot for authors with most books.
def best_authors(df):
    book_author=df.groupby('authors')['title'].count().reset_index().sort_values('title',ascending=False).head(10).set_index('authors')
    plt.figure(figsize=(15,10))
    ax=sns.barplot(book_author['title'],book_author.index,palette="inferno")
    ax.set_title("Authors with most books: ")
    ax.set_xlabel("Total Number of Books: ")
    plt.savefig('static/images/best_authors.png')


# Barpot for categories with most number of books.
def best_category(df):
    book_category=df.groupby('categories')['title'].count().reset_index().sort_values('title',ascending=False).head(10).set_index('categories')
    plt.figure(figsize=(15,10))
    ax=sns.barplot(book_category['title'],book_category.index,palette="inferno")
    ax.set_title("Categories with most books: ")
    ax.set_xlabel("Total Number of Books: ")
    plt.savefig('static/images/best_category.png')

# Barplot for year with most publisition
def best_year(df):
    df.published_year=df.published_year.astype('string')
    book_year=df.groupby('published_year')['title'].count().reset_index().sort_values('title',ascending=False).head(10).set_index('published_year')
    plt.figure(figsize=(15,10))
    ax=sns.barplot(book_year['title'],book_year.index,palette="inferno")
    ax.set_title("Year with most publisition: ")
    ax.set_xlabel("Total Number of Books: ")
    plt.savefig('static/images/best_year.png')


# Displot for average rating distribution for all books
def avg_rating_distribution(df):
    fig, ax= plt.subplots(figsize=[15,10])
    sns.distplot(df['average_rating'], ax=ax)
    ax.set_title('Average rating distribution for all books')
    ax.set_xlabel('average rating')
    plt.savefig('static/images/avg_rating_distribution.png')



app = Flask(__name__)
@app.route('/', methods =["GET", "POST"])
@app.route("/home",methods =["GET", "POST"])

def system():
    names=books_names(df)
    if request.method == "POST" and request.form.get("fname") != "": 
       book_name = request.form.get("fname") #Name of a book getting from form.
       rec = recommendations(book_name)
       my_book= "You selected:  "+book_name
       message="Our Recommendations Are : "
       return render_template("home.html",rec=rec, names=names,message=message,my_book=my_book)
    else:
        return render_template("home.html",names=names)



@app.route("/user_guide")

def user_guide():
    return render_template("user_guide.html")

@app.route("/charts")

def charts():
    best_authors(df)
    best_category(df)
    best_year(df)
    avg_rating_distribution(df)
    return render_template("charts.html")

if __name__ == '__main__':
   app.run()