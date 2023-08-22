#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
import nltk
import re


# In[3]:


df=pd.read_csv("E:/NLP DATASETS/barbie_Cleaned.csv")
df


# In[44]:


df.rating.unique()


# In[46]:


df["rating"] = df["rating"].replace(["H","B","P","M","D","T","A"],[10,0,0,0,0,0,0])


# In[47]:


df.head()


# In[48]:


df.tail()


# In[49]:


df.info()


# In[50]:


df.dtypes


# In[51]:


df.describe()


# In[52]:


df.isna().sum()


# In[53]:


df.duplicated().sum()


# In[54]:


nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")


# In[55]:


review=df["text"]
review


# In[56]:


from nltk import TweetTokenizer
tk=TweetTokenizer()
review = review.apply(lambda x:tk.tokenize(x)).apply(lambda x:" ".join(x))
review


# In[57]:


review = review.str.replace("[^a-zA-Z0-9]+"," ")
review


# In[58]:


from nltk.tokenize import word_tokenize
review = review.apply(lambda x:' '.join([w for w in word_tokenize(x) if len(w)>=3]))
review


# In[59]:


from nltk.stem import SnowballStemmer
stemmer = SnowballStemmer('english')
review = review.apply(lambda x:[stemmer.stem(i.lower()) for i in tk.tokenize(x)]).apply(lambda x:' '.join(x))
review


# In[60]:


from nltk.corpus import stopwords
nltk.download
stop=stopwords.words('english')
review = review.apply(lambda x:[i for i in word_tokenize(x) if i not in stop]).apply(lambda x:' '.join(x))
review


# In[61]:


from sklearn.feature_extraction.text import TfidfVectorizer
vec = TfidfVectorizer()
data = vec.fit_transform(review)
data


# In[62]:


data.shape


# In[63]:


X=data.toarray()


# In[78]:


df["rating"] = df["rating"].astype(int)


# In[79]:


y=df["rating"].values
y


# In[80]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.30,random_state=42)
X_train.shape,X_test.shape


# In[81]:


y_train.shape,y_test.shape


# In[82]:


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(X_train)
X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)


# In[88]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
from sklearn.metrics import classification_report,accuracy_score


# In[92]:


knn = KNeighborsClassifier(n_neighbors=7)
base = GaussianNB()
svm = SVC()
dec = DecisionTreeClassifier(criterion="entropy")
rfc = RandomForestClassifier(n_estimators=16,random_state=42)
xgb = XGBClassifier()
adb = AdaBoostClassifier(n_estimators=50, random_state=42)
lst_model = [knn,base,svm,dec,rfc,xgb,adb]


# In[93]:


for i in lst_model:
    print(i)
    print("-"*75)
    i.fit(X_train,y_train)
    y_pred=i.predict(X_test)
    print(classification_report(y_test,y_pred))
    print("Accuracy score of",i," ",accuracy_score(y_test,y_pred))
    labels=[ ]
    result=confusion_matrix(y_test,y_pred)
    cmd=ConfusionMatrixDisplay(result,display_labels=labels)
    cmd.plot()


# In[ ]:





# In[ ]:





# In[ ]:




