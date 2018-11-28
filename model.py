
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

class Model:
	"""docstring for Moddel"""
	def __init__(self):
		self.classifier = LogisticRegression(random_state = 0)
		self.lemmatizer = WordNetLemmatizer()
		self.enc        = LabelEncoder()
		self.tfidf      = TfidfVectorizer()

	def data_scoring(self,path):
		self.dataset = pd.read_csv(path)# read the orignal file of the dataset

	def clean_text (self , text):
		text = str(text) #read the line as string 
		text = text.replace(","," ") #replace the commas with the space 
		text = re.sub('[^a-zA-Z]' ,' ',text)#take only words in english language , ignore other items as numbers and other languages  
		text = text.lower() # transform to lower case 
		text = text.split() 
		text = [self.lemmatizer.lemmatize(word) for word in text if not word in set (stopwords.words('english'))]
		text = [self.lemmatizer.lemmatize(word,pos='v') for word in text ]#make lemmatize to all words to return to  their orignal form 
		text = ' '.join(text)# transform from list to string 
		return text

	def data_preprocessing(self):
		self.dataset["label"] 		=self.enc.fit_transform(self.dataset["label"])#make label encoder for the label to transfrom it to (0,1)
		self.dataset.drop("Unnamed: 0",axis=1,inplace=True)
		#self.dataset['title'] 		= self.dataset['title'].apply(self.clean_text)
		self.dataset['text'] 		= self.dataset['text'].apply(self.clean_text)# apply the clean function to the columns to be cleaned 
		self.dataset.dropna(inplace=True)# remove the rows with nan values 
		#self.dataset.to_csv("preprocessed.csv", encoding='utf-8', index=False)
		self.x 						= self.dataset['text'] #split the data to (x, y) varaibles 
		self.y 						= self.dataset['label']
		self.tfidf_x				= self.tfidf.fit_transform(self.x) # create the csr matrix for the tfidf
		self.x_train,self.x_test,self.y_train,self.y_test = train_test_split(self.tfidf_x,self.y,test_size=0.25) # split the total matrix with the label to train and test to build the model 
	
		
	def train(self,path):
		t1 = time.time() # reset the time == calculate the inital values of the timer
		t2 = time.time() # reset the time == calculate the inital values of the timer
		self.data_scoring(path) 
		print("Data Scoring take : ",(time.time()-t1)) #calculate time for only read the file 
		t1 = time.time()
		self.data_preprocessing() # call the function to prebrocessing 
		print("Data Preprocessing take : ",(time.time()-t1))
		t1 = time.time()#calculate time for only preprocess  the file
		self.classifier.fit(self.x_train,self.y_train) # split the total matrix with the label to train and test to build the model
		print("Training took     : ",(time.time()-t1))
		print("total taken time  : ",(time.time()-t2)) #calculate total time working on  file  
		#with open("model.pickle","w") as file:
			#pickle.dump(self.classifier, file)

	def evaluate(self):
		train_score = self.classifier.score(self.x_train,self.y_train) # scoring the train data to know the percent of total true classified 
		test_score  = self.classifier.score(self.x_test,self.y_test) # scoring the test data to know the percent of total true classified 
		#y_pred      = self.classifier.predict(self.x_test)
		#print(classification_report(self.y_test, y_pred))
		return train_score,test_score

	def test(self,title):
		#with open("model.pickle","w") as file:
			#self.classifier = pickle.load(file)
		title = self.clean_text(title) # apply the function to the test entered by the user 
		title = self.tfidf.transform([title]) # apply matrix by tf-idf to the test entered by the user
		print ("#########################           ########################")
		
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




