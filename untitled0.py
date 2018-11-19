import pandas as pd 
import numpy as np 
import re 
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer 
from nltk.stem import WordNetLemmatizer

df = pd.read_csv('fake_or_real_news.csv')


"""
title =re.sub('[^a-zA-Z]' ,' ', df['title'][0])
title = title.lower()
title = title.split()
ps = PorterStemmer ()
title = [ps.stem(word) for word in title if not word in set (stopwords.words('english'))]
title = ' '.join(title)"""

ps = PorterStemmer ()
lemmatizer = WordNetLemmatizer()

def clean_text (text):
    title =re.sub('[^a-zA-Z]' ,' ', text)
    title = title.lower()
    title = title.split()
    title = [lemmatizer.lemmatize(word) for word in title if not word in set (stopwords.words('english'))]
    title = ' '.join(title)
    return title
	
df['new_column'] = df['title'].apply(clean_text)   

         
        
            
    