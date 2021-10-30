import bcrypt
from django.shortcuts import redirect


import pandas as pd
import pickle
import pandas as pd
import numpy as np
import pickle
import sys
import os
import io
import re
from sys import path
import numpy as np
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
from string import punctuation, digits
from IPython.core.display import display, HTML
from nltk.corpus import stopwords
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
import pandas as pd

# For the Stop Words
from nltk.corpus import stopwords

# Convert the Words into Count Vectpr
from sklearn.feature_extraction.text import CountVectorizer

# Used to Pipe line
from sklearn.feature_extraction.text import TfidfTransformer

# Train Test Split
from sklearn.model_selection import train_test_split

# Model
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression,LinearRegression

# For Report
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

from sklearn.svm import SVC

from sklearn.svm import LinearSVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import Perceptron
from sklearn.neighbors import NearestCentroid
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import AdaBoostClassifier

from flask import Flask, render_template, url_for, request, flash, session
import mysql.connector
import pandas as pd
import string
#import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
from flask_mysqldb import MySQL
from testcode.Lib import os

app = Flask(__name__)

app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'urdoc'

mysql = MySQL(app)



@app.route('/')
def home():
    return render_template('index.html')


@app.route('/pharmacylogin')
def pharmacylogin():
    return render_template('newpharmacylogin.html')
@app.route('/adminlogin')
def adminlogin():
    return render_template('newadminlogin.html')


@app.route('/recomend')
def recomend():
    return render_template('newrecommendation.html')

@app.route('/pricepred')
def pricepred():
    return render_template('price.html')

@app.route('/review')
def review():
    return render_template('newreview.html')

@app.route('/signup',methods=['POST'])
def signup():
	if request.method == "POST":
		username = request.form['username']
		password = request.form['password'].encode('UTF-8')
		#repeatpass = bcrypt.hashpw(password,bcrypt.gensalt())
		email = request.form['email']

		cur = mysql.connection.cursor()
		cur.execute("INSERT INTO signup(username,password,email) VALUES (%s,%s,%s)", (username, password,email))
		mysql.connection.commit()

	return render_template("newpharmacistsinterface.html")



@app.route('/signin',methods=['POST'])
def signin():
	if request.method == "POST":
		username = request.form['username']
		password = request.form['password'].encode('UTF-8')
		#repeatpass = bcrypt.hashpw(password,bcrypt.gensalt())


		# search by author or book
		cur = mysql.connection.cursor()
		cur.execute("INSERT INTO signin(username,password) VALUES (%s,%s)", (username, password))
		mysql.connection.commit()

	return render_template("newpharmacistsinterface.html")





@app.route('/display',methods=['POST'])
def display():

    if request.method == "POST":
        disease = request.form['disease']
        ddi = request.form['ddi']
        age = request.form['age']
        gender = request.form.get('gender')

        cur = mysql.connection.cursor()
        #cur.execute("(SELECT DISTINCT drugname,rating,InteractDrug,disease from druginfo WHERE disease LIKE %s GROUP BY drugname"
          #         " EXCEPT"
         #          "(SELECT DISTINCT drugname,rating,InteractDrug,disease  from druginfo WHERE InteractDrug NOT LIKE %s GROUP BY drugname )ORDER BY rating DESC LIMIT 10)",(disease,ddi))
        cur.execute("SELECT DISTINCT drugname,MAX(totalrating)totalrating from druginfo WHERE disease LIKE %s AND InteractDrug NOT LIKE %s OR age LIKE %s OR gender LIKE %s GROUP BY drugname ORDER BY totalrating DESC LIMIT 20 ",(disease,ddi,age,gender))
        mysql.connection.commit()

        data = cur.fetchall()
        # all in the search box will return all the tuples
        if len(data) == 0 and disease == 'all':
            cur.execute("SELECT drugname, totalrating from druginfo")
            mysql.connection.commit()


            data = cur.fetchall()
        return render_template('newdisplay.html', data=data)
    return render_template('newdisplay.html')
    #return render_template('display.html')


@app.route('/adminlogindb',methods=['POST'])
def adminlogindb():
	if request.method == "POST":
		username = request.form['username']
		password = request.form['password'].encode('UTF-8')
		#repeatpass = bcrypt.hashpw(password,bcrypt.gensalt())


		# search by author or book
		cur = mysql.connection.cursor()
		cur.execute("INSERT INTO signin(username,password) VALUES (%s,%s)", (username, password))
		mysql.connection.commit()

	return render_template("newAdminpanel.html")




@app.route('/adminpaneldb',methods=['POST'])
def adminpaneldb():
	if request.method == 'POST':
		drugname=request.form['name']
		comment = request.form['review']
		se = request.form['sideeffects']
		#effectiveness=request.form.get('option', type=int)
		cur = mysql.connection.cursor()

		# cur.execute("(SELECT DISTINCT drugname,rating,InteractDrug,disease from druginfo WHERE disease LIKE %s GROUP BY drugname"
		#         " EXCEPT"
		#          "(SELECT DISTINCT drugname,rating,InteractDrug,disease  from druginfo WHERE InteractDrug NOT LIKE %s GROUP BY drugname )ORDER BY rating DESC LIMIT 10)",(disease,ddi))
		cur.execute("INSERT INTO adminpanel(drugname,review,sideeffect) VALUES (%s,%s,%s)", (drugname, comment, se))

		mysql.connection.commit()

		#data = cur.fetchall()
	return render_template('newadminpanel.html')

@app.route('/addadmininputs',methods=['POST'])
def addadmininputs():
	if request.method == 'POST':
		name=request.form['name']
		se = request.form['sideeffects']
		des = request.form['review']

		#effectiveness=request.form.get('option', type=int)
		cur = mysql.connection.cursor()

		# cur.execute("(SELECT DISTINCT drugname,rating,InteractDrug,disease from druginfo WHERE disease LIKE %s GROUP BY drugname"
		#         " EXCEPT"
		#          "(SELECT DISTINCT drugname,rating,InteractDrug,disease  from druginfo WHERE InteractDrug NOT LIKE %s GROUP BY drugname )ORDER BY rating DESC LIMIT 10)",(disease,ddi))
		cur.execute("INSERT INTO adminpanel(drugname,review,sideeffect) VALUES (%s,%s,%s)", (name, des, se))

		mysql.connection.commit()

		#data = cur.fetchall()
	return render_template('MedAdddMesg.html')

@app.route('/predict',methods=['POST'])
def predict():
	df= pd.read_csv("Reviews.csv")
	data = df[['rating', 'verified_reviews']]

	data['index'] = data.index
	# Text Preprocessing
	columns = ['index', 'rating', 'verified_reviews']
	df_ = pd.DataFrame(columns=columns)

	# remove numbers
	data['verified_reviews'] = data['verified_reviews'].replace('\d', '', regex=True)
	# lower string
	data['verified_reviews'] = data['verified_reviews'].str.lower()
	# remove email adress
	data['verified_reviews'] = data['verified_reviews'].replace('[a-zA-Z0-9-_.]+@[a-zA-Z0-9-_.]+', '', regex=True)
	# remove IP address
	data['verified_reviews'] = data['verified_reviews'].replace('((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)(\.|$)){4}', '',
																regex=True)
	# remove punctaitions and special chracters
	data['verified_reviews'] = data['verified_reviews'].str.replace('[^\w\s]', '')
	# reving URLs
	data['verified_reviews'] = data['verified_reviews'].replace('https?://\S+|www\.\S+', '', regex=True)

	X_Data = data['verified_reviews']
	Y_Data = data['rating']
	# countvecotizor
	cv = CountVectorizer(ngram_range=(1, 1),
						 stop_words=['once', 'each', 'below', 'do', 'up', 'doing', 'under', 'only', 'against', 'during',
									 'why', 'and', 'other', 'how', 'll', 'before', 'now', 'his', "you're", 'off',
									 'where', 'out', 'he', 'then', 'or', 'no', 'me', 'those', 'them', 'are', 'they',
									 'but', 'herself', 'it', "that'll", "you'll", 'be', 'ours', 'whom', 'which', 'into',
									 'a', 'with', 'after', 'will', 'while', 'about', 'has', 'both', 're', 'as', 'few',
									 'too', 'such', 'own', 'does', 'for', 'yourselves', 'my', 'in', 'you', 'ma',
									 'ourselves', 'weren', 'again', 'on', 'hers', 'any', 'nor', 'through', 'is', 'same',
									 'this', 've', 'when', 'most', 'if', 'just', 'so', 'very', 'can', 'was', 'being',
									 "you'd", 'myself', 'himself', 'from', 'themselves', 'to', 'am', 'the', 'him',
									 'all', 'because', 'we', 'than', 'what', 'there', 'itself', 'having', 'until',
									 'here', 'at', 'that', 'further', 'theirs', 'd', "she's", 'm', 'more', 'their',
									 "you've", 'between', 'above', 'some', 'yours', 'an', 'its', 'who', 'been', 'over',
									 'our', 'o', 'did'])
	X_Data = cv.fit_transform(X_Data)
	SEED = 2000
	X_Train, X_Test, Y_Train, Y_Test = train_test_split(X_Data, Y_Data, test_size=0.3, random_state=SEED)
	# classification model
	from sklearn.naive_bayes import MultinomialNB

	model = MultinomialNB()
	model.fit(X_Train, Y_Train)
	#predicted = model.predict(X_Test,)
	#accuracy = accuracy_score(Y_Test, predicted)


	#Multimonial Naive Bayes Classifier
	from sklearn.naive_bayes import MultinomialNB
	#clf = MultinomialNB()
	#clf.fit(X_train,y_train)
	#clf.score(X_test,y_test)
	#Alternative Usage of Saved Model
	# ytb_model = open("naivebayes_spam_model.pkl","rb")
	# clf = joblib.load(ytb_model)



	if request.method == 'POST':
		drugname=request.form['name']
		review = request.form['review']
		se = request.form['sideeffects']
		effectiveness=request.form.get('option', type=int)
		reviewarray = [review]
		searray=[se]

		#cur.close()
		#return 'success'
		vect = cv.transform(reviewarray).toarray()
		pred = model.predict(vect)
		vact=cv.transform(searray).toarray()
		predii=model.predict(vact)
		# database
		cur = mysql.connection.cursor()
		my_prediction=predii+pred+effectiveness

		cur.execute("INSERT INTO feedback(name,review,sideeffect,sideeffectscore,star,feedbackstars) VALUES (%s,%s,%s,%s,%s,%s)", (drugname, review, se,predii,my_prediction,pred))
		cur.execute("Update feedback Set editedstars=REPLACE(REPLACE(star,'[',''),']','')")
		cur.execute("UPDATE feedback SET m = (SELECT AVG(editedstars)FROM feedback WHERE name LIKE %s)WHERE name LIKE %s",[drugname,drugname])
		cur.execute("UPDATE druginfo SET feedbackrating = %s WHERE drugname LIKE %s", [my_prediction,drugname])
		#cur.execute("UPDATE feedback SET feedbackrating = %s WHERE drugname LIKE %s", [pred, name])
		cur.execute("UPDATE druginfotest SET a = (SELECT Distinct m FROM feedback WHERE name LIKE %s)WHERE drugname LIKE %s", [drugname,drugname])
		cur.execute("UPDATE druginfo SET feedbackrating = (SELECT Distinct m FROM feedback WHERE name LIKE %s)WHERE drugname LIKE %s",[drugname, drugname])
		cur.execute("UPDATE druginfo SET totalrating = rating + feedbackrating")
		# cur.execute("SELECT Distinct drugname,rating from druginfo WHERE drugname LIKE %s ORDER BY rating DESC LIMIT 1", (name,))
		# cur.execute("INSERT INTO dbtest(drugname,rating) VALUES (%s,%s) WHERE drugname %s", (name, my_prediction))
		# cur.execute("INSERT INTO dbtest(rating) SELECT rating FROM druginfo WHERE drugname LIKE %s ", [name,])
		mysql.connection.commit()
	return render_template('newresult.html',prediction = pred,p=predii)

if __name__ == '__main__':
	app.debug = True
	#port = int(os.environ.get('PORT', 5000))
	app.run()