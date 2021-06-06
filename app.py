#Import Library
from flask import Flask, render_template, request, url_for, send_file
from flask_caching import Cache
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from textvec import vectorizers
from textvec.vectorizers import TfrfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import os
import tweepy
import csv
import joblib
import nltk
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
from sentimen import get_sentiment
from preprocessing import *

# =============================TWITTER AUTHENTICATION======================
consumer_key = "JRwri98iEGbQHbwTz6i75gMEb"
consumer_secret = "w4WySeAYuMRBYJdq3DwmZjW0MeC1uC3rLfsLg5lcubtvBGsMIU"
access_token = "1263049310536527872-ZuO7lZz8ewOa12vGJNfSCEcZzSeEcn"
access_token_secret = "yNkp5JcEsRPrZXC8zISHSb6BpgmSAhqFB8H7MLcWsD7zf"

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth,wait_on_rate_limit=True)
# =========================================================================

cache = Cache()

app = Flask(__name__)

app.config["CACHE_TYPE"] = "null"
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

cache.init_app(app)

@app.route("/", methods=["POST", "GET"])
def index():
    if request.method == "POST":
        content = request.form.get("Name")
        sentiment = get_sentiment(content)
        return render_template("index.html", sentiment = sentiment)
    return render_template("index.html")

@app.route("/crawling", methods=["POST", "GET"])
def crawling():
    if request.method == "GET":
        return render_template("crawling.html")

    elif request.method == "POST":
        cache.clear()
        keyword = request.form.get("data")

        header = ['Created At', 'Screen Name', 'Full Tweet']
        with open("static/data/tweet.csv", "w", newline="", encoding="utf-8") as f:
            csvWriter = csv.writer(f)
            csvWriter.writerow(header)
            for tweet in tweepy.Cursor(api.search, q= keyword + "-filter:retweets", lang="id", tweet_mode="extended").items(100):
                csvWriter.writerow([tweet.created_at,tweet.user.name, tweet.full_text.encode('utf-8')])

        df = pd.read_csv('static/data/tweet.csv')

        #drop kolom yang tidak diperlukan
        df = df.drop(columns=['Created At', 'Screen Name'])
        #menghapus duplikat tweet
        df = df.drop_duplicates()

        df['Cleaning'] = df['Full Tweet'].apply(lambda x: cleaning(x))
        df['Case Folding'] = df['Cleaning'].apply(lambda x: case_folding(x))
        df['Tweet Bersih'] = df['Case Folding'].apply(lambda x: bersih(x))
        
        X_test = df['Tweet Bersih']
        loaded_model = joblib.load('clf_rf_rbf.pkl')
        df['Pred'] = loaded_model.predict(X_test)

        label = []
        for index, row in df.iterrows():
            if row["Pred"] == 1:
                label.append("Positif")
            elif row["Pred"] == -1:
                label.append("Negatif")
            else:
                label.append("Netral")

        df["Label"] = label

        print(df['Pred'].value_counts())
        
        if os.path.exists("static/image/Chart.png"):
            os.remove("static/image/Chart.png")

        label = ["Positif", "Netral", "Negatif"]
        exp = [0, 0, 0.1]
        plt.pie(df['Pred'].value_counts(), labels=label, explode=exp, autopct='%2.0f%%')
        plt.savefig('static/image/Chart.png', bbox_inches='tight', transparent=True, pad_inches=0.5)
        plt.close()

        gk = df.groupby('Label')

        return render_template('hasil_prediksi.html', keyword=keyword, gk=gk.first().to_html())

@app.route("/upload", methods=["POST", "GET"])
def upload():
    if request.method == "GET":
        return render_template("upload.html")

    elif request.method == "POST":
        file_upload = request.files.get("file")
        # print(file_upload)
        tweet = pd.read_csv(file_upload)
        tweet.to_csv('static/data/fileUpload.csv', index=False)
        data = tweet.head().to_html(index=False)
        return render_template("hasil_upload.html", data = data)

@app.route("/preprocessing", methods=["POST", "GET"])
def preprocessing():
    if request.method == "POST":
        df = pd.read_csv('static/data/fileUpload.csv')

        #Preprocessing
        df = df.drop_duplicates()
        df['Cleaning'] = df['Teks'].apply(lambda x: cleaning(x))
        df['Case Folding'] = df['Cleaning'].apply(lambda x: case_folding(x))
        df['Tweet Bersih'] = df['Case Folding'].apply(lambda x: bersih(x))

        label = []
        for index, row in df.iterrows():
            if row["Label"] == "Positif":
                label.append(1)
            elif row["Label"] == "Negatif":
                label.append(-1)
            else:
                label.append(0)
        df["Label"] = label
        df.to_csv('static/data/hasil_preprocessing.csv', index=False)
        return render_template("hasil_preprocessing.html", df = df.head().to_html(index=False))

@app.route("/prediksi", methods=["POST", "GET"])
def prediksi():
    if request.method == "POST":
        cache.clear()
        df_train = pd.read_csv('static/data/data_train.csv')
        df_test = pd.read_csv('static/data/hasil_preprocessing.csv')

        X_train = df_train["Tweet Bersih"]
        y_train = df_train["Label"].values
        X_test = df_test["Tweet Bersih"]
        y_test = df_test["Label"].values

        clf_idf_poly = joblib.load('clf_idf_poly.pkl')
        pred1 = clf_idf_poly.predict(X_test)
        print(classification_report(pred1, y_test))
        akurasi1 = accuracy_score(pred1, y_test)
        akurasi1 = round(akurasi1, 2)*100
        akurasi1 = round(akurasi1)
        f1 = f1_score(pred1, y_test, average='macro')
        f1 = round(f1, 2)*100
        f1 = round(f1)

        clf_idf_rbf = joblib.load('clf_idf_rbf.pkl')
        pred2 = clf_idf_rbf.predict(X_test)
        print(classification_report(pred2, y_test))
        akurasi2 = accuracy_score(pred2, y_test)
        akurasi2 = round(akurasi2, 2)*100
        akurasi2 = round(akurasi2)
        f2 = f1_score(pred2, y_test, average='macro')
        f2 = round(f2, 2)*100
        f2 = round(f2)

        clf_rf_poly = joblib.load('clf_rf_poly.pkl')
        pred3 = clf_rf_poly.predict(X_test)
        print(classification_report(pred3, y_test))
        akurasi3 = accuracy_score(pred3, y_test)
        akurasi3 = round(akurasi3, 2)*100
        akurasi3 = round(akurasi3)
        f3 = f1_score(pred3, y_test, average='macro')
        f3 = round(f3, 2)*100
        f3 = round(f3)

        clf_rf_rbf = joblib.load('clf_rf_rbf.pkl')
        pred4 = clf_rf_rbf.predict(X_test)
        print(classification_report(pred4, y_test))
        akurasi4 = accuracy_score(pred4, y_test)
        akurasi4 = round(akurasi4, 2)*100
        akurasi4 = round(akurasi4)
        f4 = f1_score(pred4, y_test, average='macro')
        f4 = round(f4, 2)*100
        f4 = round(f4)

        if os.path.exists("static/data/report1.csv"):
            os.remove("static/data/report1.csv")
            os.remove("static/data/report2.csv")
            os.remove("static/data/report3.csv")
            os.remove("static/data/report4.csv")

        report1 = pd.DataFrame(classification_report(pred1, y_test, output_dict=True)).transpose()
        report1.to_csv('static/data/report1.csv', index= True)
        df_report1 = pd.read_csv('static/data/report1.csv')

        report2 = pd.DataFrame(classification_report(pred2, y_test, output_dict=True)).transpose()
        report2.to_csv('static/data/report2.csv', index= True)
        df_report2 = pd.read_csv('static/data/report2.csv')

        report3 = pd.DataFrame(classification_report(pred3, y_test, output_dict=True)).transpose()
        report3.to_csv('static/data/report3.csv', index= True)
        df_report3 = pd.read_csv('static/data/report3.csv')

        report4 = pd.DataFrame(classification_report(pred4, y_test, output_dict=True)).transpose()
        report4.to_csv('static/data/report4.csv', index= True)
        df_report4 = pd.read_csv('static/data/report4.csv')

        if os.path.exists("static/image/confusion_matrix1.png"):
            os.remove("static/image/confusion_matrix1.png")
            os.remove("static/image/confusion_matrix2.png")
            os.remove("static/image/confusion_matrix3.png")
            os.remove("static/image/confusion_matrix4.png")

        # Confusion matrix
        matrix1 = plot_confusion_matrix(clf_idf_poly, X_test, y_test, values_format='d')
        matrix1.ax_.set_title('Confusion Matrix TF-IDF & Polynomial Kernel', color='black')
        plt.xlabel('Predicted Label', color='black')
        plt.ylabel('True Label', color='black')
        plt.gcf().axes[0].tick_params(color='black')
        plt.gcf().axes[1].tick_params(color='black')
        plt.gcf().set_size_inches(8,4)
        plt.savefig('static/image/confusion_matrix1.png', bbox_inches='tight', transparent=True, pad_inches=0.5)
        plt.close()

        matrix2 = plot_confusion_matrix(clf_idf_rbf, X_test, y_test, values_format='d')
        matrix2.ax_.set_title('Confusion Matrix TF-IDF & RBF Kernel', color='black')
        plt.xlabel('Predicted Label', color='black')
        plt.ylabel('True Label', color='black')
        plt.gcf().axes[0].tick_params(color='black')
        plt.gcf().axes[1].tick_params(color='black')
        plt.gcf().set_size_inches(8,4)
        plt.savefig('static/image/confusion_matrix2.png', bbox_inches='tight', transparent=True, pad_inches=0.5)
        plt.close()

        matrix3 = plot_confusion_matrix(clf_rf_poly, X_test, y_test, values_format='d')
        matrix3.ax_.set_title('Confusion Matrix TF-RF & Polynomial Kernel', color='black')
        plt.xlabel('Predicted Label', color='black')
        plt.ylabel('True Label', color='black')
        plt.gcf().axes[0].tick_params(color='black')
        plt.gcf().axes[1].tick_params(color='black')
        plt.gcf().set_size_inches(8,4)
        plt.savefig('static/image/confusion_matrix3.png', bbox_inches='tight', transparent=True, pad_inches=0.5)
        plt.close()

        matrix4 = plot_confusion_matrix(clf_rf_rbf, X_test, y_test, values_format='d')
        matrix4.ax_.set_title('Confusion Matrix TF-RF & RBF Kernel', color='black')
        plt.xlabel('Predicted Label', color='black')
        plt.ylabel('True Label', color='black')
        plt.gcf().axes[0].tick_params(color='black')
        plt.gcf().axes[1].tick_params(color='black')
        plt.gcf().set_size_inches(8,4)
        plt.savefig('static/image/confusion_matrix4.png', bbox_inches='tight', transparent=True, pad_inches=0.5)
        plt.close()

        os.remove("static/data/fileUpload.csv")
        os.remove("static/data/hasil_preprocessing.csv")

        return render_template("hasil_classification.html", df_report1 = df_report1.to_html(index=False),
        akurasi1 = akurasi1, f1=f1,akurasi2 = akurasi2, f2=f2,akurasi3 = akurasi3, f3=f3,akurasi4 = akurasi4,
        f4=f4, df_report2 = df_report2.to_html(index=False), df_report3 = df_report3.to_html(index=False), 
        df_report4 = df_report4.to_html(index=False))

@app.route("/about", methods=["POST", "GET"])
def about():
    return render_template("about.html")

if __name__ == '__main__':
    app.run(debug = True, port = 5000)