import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.externals import joblib
import re 
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

class Model:
    lemmatizer = WordNetLemmatizer()
    def __init__(self):
        TFIDF_title        = 'model/tfidf_title.sav'
        model_title        = 'model/model_title.sav'
        TFIDF_text         = 'model/tfidf_text.sav'
        model_text         = 'model/model_text.sav'

        self.tfidf_title       = joblib.load(TFIDF_title)
        self.classifier_title  = joblib.load(model_title)
        self.tfidf_text        = joblib.load(TFIDF_text)
        self.classifier_text   = joblib.load(model_text)

    def clean_text (self , text):
        text = str(text)
        text = text.replace(","," ")
        text = re.sub('[^a-zA-Z]' ,' ',text)
        text = text.lower()
        text = text.split()
        text = [self.lemmatizer.lemmatize(word) for word in text if not word in set (stopwords.words('english'))]
        text = [self.lemmatizer.lemmatize(word,pos='v') for word in text ]
        text = ' '.join(text)
        return text

    def test_title(self,title):
        title = self.clean_text(title)
        title = self.tfidf_title.transform([title])
        return  self.classifier_title.predict(title)

    def test_article(self,acticle):
        acticle = self.clean_text(acticle)
        acticle = self.tfidf_text.transform([acticle])
        return    self.classifier_text.predict(acticle)
