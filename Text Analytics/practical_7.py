'''
Text Analytics
1.	Extract Sample document and apply following document preprocessing methods: Tokenization, POS Tagging, stop words removal, Stemming and Lemmatization.
2.	Create representation of document by calculating Term Frequency and Inverse Document
Frequency.
'''

import nltk
from nltk.tokenize import word_tokenize
#function to split text into word
nltk.download('punkt')

tokens = word_tokenize("The quick brown fox jumps over the lazy dog")
nltk.download('stopwords')
print(tokens)


from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
tokens = [w for w in tokens if not w in stop_words]
print(tokens)

#NLTK provides several stemmer interfaces like Porter stemmer, #Lancaster Stemmer, Snowball Stemmer
from nltk.stem.porter import PorterStemmer
porter = PorterStemmer()
stems = []
for t in tokens:    
    stems.append(porter.stem(t))
print(stems)



