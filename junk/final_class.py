
import re 
import time
import pandas as pd 
import numpy as np 
import pickle
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer 
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
#ahmed = pd.read_csv('fake_or_real_news.csv')

class Model:
	"""docstring for Moddel"""
	def __init__(self):
		self.classifier = LogisticRegression(random_state = 0)
		self.lemmatizer = WordNetLemmatizer()
		self.enc        = LabelEncoder()
		self.tfidf      = TfidfVectorizer()

	def data_scoring(self,path):
		self.dataset = pd.read_csv(path)

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

	def data_preprocessing(self):
		self.dataset["label"] 		=self.enc.fit_transform(self.dataset["label"])
		self.dataset.drop("Unnamed: 0",axis=1,inplace=True)
		#self.dataset['title'] 		= self.dataset['title'].apply(self.clean_text)
		self.dataset['text'] 		= self.dataset['text'].apply(self.clean_text)
		self.dataset.dropna(inplace=True)
		#self.dataset.to_csv("preprocessed.csv", encoding='utf-8', index=False)
		self.x 						= self.dataset['text']
		self.y 						= self.dataset['label']
		self.tfidf_x				= self.tfidf.fit_transform(self.x)
		self.x_train,self.x_test,self.y_train,self.y_test = train_test_split(self.tfidf_x,self.y,test_size=0.25)
	
		
	def train(self,path):
		t1 = time.time()
		t2 = time.time()
		self.data_scoring(path)
		print("Data Scoring take : ",(time.time()-t1))
		t1 = time.time()
		self.data_preprocessing()
		print("Data Preprocessing take : ",(time.time()-t1))
		t1 = time.time()
		self.classifier.fit(self.x_train,self.y_train)
		print("Training took     : ",(time.time()-t1))
		print("total taken time  : ",(time.time()-t2))
		#with open("model.pickle","w") as file:
			#pickle.dump(self.classifier, file)

	def evaluate(self):
		train_score = self.classifier.score(self.x_train,self.y_train)
		test_score  = self.classifier.score(self.x_test,self.y_test)
		#y_pred      = self.classifier.predict(self.x_test)
		#print(classification_report(self.y_test, y_pred))
		return train_score,test_score

	def test(self,title):
		#with open("model.pickle","w") as file:
			#self.classifier = pickle.load(file)
		title = self.clean_text(title)
		title = self.tfidf.transform([title])
		print ("#########################           ########################")
		"""
		feature_names = self.tfidf.get_feature_names()
		print(type(feature_names))
		print(type(title.split()))
		for i in list(title.split()): 
			if i in list(feature_names):
				print(i," is exist")
			else:
				print(i, " not exist")
		#print (feature_names)
		"""
		print(title)
		print ("#########################           ########################")
		return self.classifier.predict(title)

if __name__ == '__main__':
	model = Model()
	model.train("fake_or_real_news.csv")
	print("=====================================================")
	print(model.evaluate())
	title = "The visit by Kerry, who has family and childhood ties to the country and speaks fluent French"\
	"could address some of the criticism that the United States snubbed France in its darkest hour in"\
    "The French press on Monday was filled with questions about why neither President Obama nor Kerry"\
	"attended Sunday’s march, as about 40 leaders of other nations did. Obama was said to have stayed "\
	#text = "no stranger to intense security, who marched beside Hollande through the city streets. The highest ranking U.S. officials attending the march were Jane Hartley, the ambassador to France, and Victoria Nuland, the assistant secretary of state for European affairs. Attorney General Eric H. Holder Jr. was in Paris for meetings with law enforcement officials but did not participate in the march."
	print("=====================================================")
	print(model.test(title))




