from flask import Flask
from flask import redirect, url_for
from flask import render_template
from flask import request,session   
import requests
from datetime import timedelta
from bs4 import BeautifulSoup

from bs4 import BeautifulSoup
import pandas as pd
import numpy as np

from time import sleep
from random import randint
from flask_table import Table, Col
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from werkzeug.utils import secure_filename
from werkzeug.datastructures import  FileStorage
import re
from PIL import Image
import gensim
from gensim import corpora

from nltk import FreqDist
import plotly
import plotly.graph_objs as go
from bokeh.io import output_file, show
from bokeh.plotting import figure
import json
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('wordnet')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')

cut_name = []
dates = []
rating = []
cust_review = []  



source = requests.get('https://webscraper.netlify.com/').text
soup = BeautifulSoup(source, 'lxml')

app= Flask(__name__)
app.secret_key = 'hello'
app.permanent_session_lifetime = timedelta(seconds=2)




def get_profile_detail(user_handle): 
  
    url = "https://www.flipkart.com/search?q={}".format(user_handle)
  
    page = requests.get(url)
    soup = BeautifulSoup(page.content,'html.parser')
    print(soup.prettify())
    final_link = []
    for a in soup.find_all('a',class_='_31qSD5', href=True):
       final_link.append("https://www.flipkart.com"+a['href'])
    
    
    
    price = soup.find_all("div",class_= "_1vC4OE _2rQ-NK")
    name = soup.find_all("div",class_= "col col-7-12")
    
    review_title = []
    price_pro = []
    for i in range(0,len(name)):
        review_title.append(name[i].get_text())
        
    review_title[:]= [name.lstrip("\n")for name in review_title]
    review_title[:]= [name.rstrip("\n")for name in review_title]
    
    for i in range(0,len(price)):
        price_pro.append(price[i].get_text())
        
    price_pro[:]= [price.lstrip("\n")for price in price_pro]
    price_pro[:]= [price.rstrip("\n")for price in price_pro]
        
       
    return review_title,price_pro,final_link


@app.route('/<user_handle>/') 
def home(user_handle):
    review_title,price_pro,final_link = get_profile_detail(user_handle)
    amazon =pd.DataFrame({'price': price_pro, 'name': review_title, 'link': final_link})
    html = amazon.to_html()
    return html 

def get_review_detail(url_handle,pd_handle): 
    header={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/78.0.3904.108 Safari/537.36'}
    cust_review = []
    html=home(pd_handle)
    dfs = pd.read_html(html)
    df = dfs[0]
    def num(url_handle):
        return f'{url_handle}'
    orange=int(num(url_handle))
    trng=df.iloc[orange,3]
    res = trng.split('srno', 1)[0] 

    reep=str(res).replace('/p/', '/product-reviews/')
    reep
    urr=reep+'page='
    
    rev_num = requests.get(trng, headers=header)
    
    soup = BeautifulSoup(rev_num.text, 'html.parser')
    list_header=[] 
    pa = soup.find_all('span', class_='_38sUEc')
    ss=str(pa).split('&amp', 1)[1]
    ss
    su=str(ss).split('Reviews', 1)[0]
    su
    l=su[15:]
    int_b = int(l.replace(',',''))
    print ("The integer value",int_b)
    srinivas=(int_b//10)+1
    srinivas
    pages = np.arange(1, srinivas)
    for items in header: 
        try: 
            list_header.append(items.get_text()) 
        except: 
            continue
  
    for page in pages: 
      
      page = requests.get(urr + str(page) + "&ref_=adv_nxt", headers=header)
    
      soup = BeautifulSoup(page.text, 'html.parser')
      
      review_div = soup.find_all('div', class_='col _390CkK _1gY8H-')
      
      sleep(randint(1,8))
      for container in review_div:
            # runtime
            review = container.find('div', class_='qwjRop').text 
            mod_review= review.rstrip("READ MORE")
            cust_review.append(mod_review)
            dataFrame = pd.DataFrame(data = cust_review) 
            dataFrame.to_csv('Geeks.csv')
    
    return cust_review



@app.route('/homme/<url_handle>/<pd_handle>/') 
def homme(url_handle,pd_handle):
    cust_review=get_review_detail(url_handle,pd_handle)
    flipkart = pd.DataFrame({'Review': cust_review})
    Review_tbe=flipkart.to_html()
    return Review_tbe 

@app.route('/download/<url_handle>/<pd_handle>/') 
def download(url_handle,pd_handle):
    cust_review=get_review_detail(url_handle,pd_handle) 
    flipkart = pd.DataFrame({'Review': cust_review})
    df= flipkart.to_html()
    return df
    
 
@app.route('/king')
def king():
    return redirect(url_for('home', user_handle='mobile'))

@app.route('/',methods=['POST','GET']) 
def login(): 
    if request.method == 'POST':
        session.permanent = True
        user = request.form['nm']
        session['user']=user
        return redirect(url_for('user'))
    else:
        if 'user' in session:
            return redirect(url_for('user'))
        return render_template('login.html')
    
    
@app.route('/user',methods=['POST','GET'])
def user(user_handle='sumit'):
    if request.method == 'POST':
        session.permanent = True
        usr = request.form['url']
        session['user']=usr
        return redirect(url_for('home', user_handle= usr))

    
    if 'user' in session:
        user = session['user']
        return render_template("user.html", user=user)
            
    else:
        return redirect(url_for('login'))
   
@app.route('/logout.html/',methods=['POST','GET'])
def logout():
    if request.method == 'POST':
        session.permanent = True
        ussr = request.form['urll']
        usssr = request.form['urlll']
        session['user']=ussr
        return render_template('redirect.html',number=ussr,pd=usssr)
    else:
        return render_template('logout.html')

@app.route('/logout1.html/')  
def upload():  
    return render_template("logout 1.html")  
 
@app.route('/success', methods = ['POST'])  
def success():  
    if request.method == 'POST':  
        f = request.files['file']  
        f.save(f.filename)  
        return render_template("success.html", name = f.filename) 

from textblob import TextBlob

def fetch_sentiment_using_textblob(text):
    sentiment = []
    for i in text: 
        analysis = TextBlob(i)
        # set sentiment 
        if analysis.sentiment.polarity >= 0:
            sentiment.append('positive')
        else: 
            sentiment.append('negative')
    return sentiment


    
@app.route('/polarity')
def polarity():
    data1=pd.read_html('datalog.html')
    df=data1[0]
    final_data=pd.DataFrame(df)
    corpus = []
    for i in range(0,len(final_data)):
        review = re.sub('[^a-zA-Z]', ' ', final_data['Review'][i])
        review = review.lower()
        review = review.split()
        ps = WordNetLemmatizer()
        review = [ps.lemmatize(word) for word in review if not word in set(stopwords.words('english'))]
        review = ' '.join(review)
        corpus.append(review)
    review_list = final_data['Review'].tolist() 
    final_data['sentiment']= fetch_sentiment_using_textblob(review_list)
    final_data['sentiment'].value_counts()
    pos=final_data.groupby('sentiment')
    pos_values=pos.get_group('positive')
    pos_values
    pos_values['Review']
    sid = SentimentIntensityAnalyzer()
    sentiment_summary = dict()
    # for readme in readmes:
    #     sentences = nltk.tokenize.sent_tokenize(readme)
    #     for sentence in sentences:
    #         sentiment_score = sid.polarity_scores(sentence)messages = pos_list
    messages = pos_values['Review']
    summary = {"positive":0,"neutral":0,"negative":0}
    for x in messages: 
        ss = sid.polarity_scores(x)
        if ss["compound"] == 0.0: 
            summary["neutral"] +=1
        elif ss["compound"] > 0.0:
            summary["positive"] +=1
        else:
            summary["negative"] +=1
    sen=[]

    sid = SentimentIntensityAnalyzer()
    sentiment_summary = dict()
    # for readme in readmes:
    #     sentences = nltk.tokenize.sent_tokenize(readme)
    #     for sentence in sentences:
    #         sentiment_score = sid.polarity_scores(sentence)messages = pos_list
    messages = pos_values['Review']
    summary = {"positive":0,"neutral":0,"negative":0}
    for x in messages: 
        ss = sid.polarity_scores(x)
        if ss["compound"] == 0.0: 
            sen.append('Neutral')
        elif ss["compound"] > 0.0:
            sen.append('Positive')
        else:
            sen.append('negative')
    pd.DataFrame(sen).count()

    pos_values.index
    
    positive=pd.concat([pos_values.drop('sentiment',axis=1),pd.DataFrame(sen,index=pos_values.index)],axis=1)
    
    pos.get_group('negative')
    
    sds=positive.rename(columns={0: "sentiment"})
    
    sumit=pd.concat([sds,pos.get_group('negative')])
    
    
    sns.countplot('sentiment',data=sumit)
    lp=sumit['sentiment'].value_counts()
    kp=pd.DataFrame(lp)
    kp.reset_index(inplace=True)
    kp['sentiment']
    output_file('bars.html')
    colors = ["#c9d9d3", "#718dbf", "#e84d60"]
    p=figure(x_range= list(kp['index']), plot_height=250, title="polarity",
           toolbar_location=None, tools="")
    p.vbar(
         x= list(kp['index']),
        top= list(kp['sentiment']),color=colors,
        width=0.9
    )

    p.xgrid.grid_line_color = None
    p.y_range.start = 0
    show(p)

    return sumit.to_html()

def par():
    data1=pd.read_html('datalog.html')
    df=data1[0]
    final_data=pd.DataFrame(df)
    corpus = []
    for i in range(0,len(final_data)):
        review = re.sub('[^a-zA-Z]', ' ', final_data['Review'][i])
        review = review.lower()
        review = review.split()
        ps = WordNetLemmatizer()
        review = [ps.lemmatize(word) for word in review if not word in set(stopwords.words('english'))]
        review = ' '.join(review)
        corpus.append(review)
    review_list = final_data['Review'].tolist() 
    final_data['sentiment']= fetch_sentiment_using_textblob(review_list)
    final_data['sentiment'].value_counts()
    pos=final_data.groupby('sentiment')
    pos_values=pos.get_group('positive')
    pos_values
    pos_values['Review']
    sid = SentimentIntensityAnalyzer()
    sentiment_summary = dict()
    # for readme in readmes:
    #     sentences = nltk.tokenize.sent_tokenize(readme)
    #     for sentence in sentences:
    #         sentiment_score = sid.polarity_scores(sentence)messages = pos_list
    messages = pos_values['Review']
    summary = {"positive":0,"neutral":0,"negative":0}
    for x in messages: 
        ss = sid.polarity_scores(x)
        if ss["compound"] == 0.0: 
            summary["neutral"] +=1
        elif ss["compound"] > 0.0:
            summary["positive"] +=1
        else:
            summary["negative"] +=1
    sen=[]

    sid = SentimentIntensityAnalyzer()
    sentiment_summary = dict()
    # for readme in readmes:
    #     sentences = nltk.tokenize.sent_tokenize(readme)
    #     for sentence in sentences:
    #         sentiment_score = sid.polarity_scores(sentence)messages = pos_list
    messages = pos_values['Review']
    summary = {"positive":0,"neutral":0,"negative":0}
    for x in messages: 
        ss = sid.polarity_scores(x)
        if ss["compound"] == 0.0: 
            sen.append('Neutral')
        elif ss["compound"] > 0.0:
            sen.append('Positive')
        else:
            sen.append('negative')
    pd.DataFrame(sen).count()

    pos_values.index
    
    positive=pd.concat([pos_values.drop('sentiment',axis=1),pd.DataFrame(sen,index=pos_values.index)],axis=1)
    
    pos.get_group('negative')
    
    sds=positive.rename(columns={0: "sentiment"})
    
    sumit=pd.concat([sds,pos.get_group('negative')])
    return sumit
@app.route('/word')
def word():
    data=par()
    wc = WordCloud(background_color="black",
                   width = 800, height = 800)
        
    wc.generate(str(data['Review']))
    wc.to_file("wordcloud.png")

    filename = Image.open("wordcloud.png")
    filename.show()
    return '<h1>see the wordcloud in the PNG FILE BELOW 👇</h1>'

@app.route('/positive')
def positive():
    data=par()
    sent=data.groupby('sentiment')
    positive_values=sent.get_group('Positive')
    wc = WordCloud(background_color="black",
                   width = 800, height = 800)
        
    wc.generate(str(positive_values['Review']))
    wc.to_file("wordcloud.png")

    filename = Image.open("wordcloud.png")
    filename.show()
    return '<h1>see the wordcloud in the PNG FILE BELOW 👇</h1>'

@app.route('/negative')
def negative():
    data=par()
    sent=data.groupby('sentiment')
    positive_values=sent.get_group('negative')
    wc = WordCloud(background_color="black",
                   width = 800, height = 800)
        
    wc.generate(str(positive_values['Review']))
    wc.to_file("wordcloud.png")

    filename = Image.open("wordcloud.png")
    filename.show()
    return '<h1>see the wordcloud in the PNG FILE BELOW 👇</h1>'

    

if __name__ == '__main__': 
    app.run(debug=True)
    

    
    app.run(host='0.0.0.0', port=5000)