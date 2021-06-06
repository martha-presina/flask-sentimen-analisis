import numpy as np
import re
import string

import nltk
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

nltk.download('stopwords')
stopword = nltk.corpus.stopwords.words('indonesian')

factory = StemmerFactory()
stemmer = factory.create_stemmer()

def cleaning(text):
  #menghapus tab, baris baru
  text = text.replace('\\t',"  ").replace('\\n',"  ").replace('\\u',"  ").replace("b'", "  ")
  # #menghapus emoticon dll
  text = text.encode('ascii','ignore').decode('ascii')
  text = re.sub(r'[^\x00-\x7f]',r'  ', text)
  #menghapus mantion dan hashtag
  text = ' '.join(re.sub("([@#][A-Za-z0-9]+)|(\w+:\/\/\S+)","  ", text).split())
  #mengapus angka
  text = re.sub(r"\d+", "  ", text)
  #menghapus karakter dibawah 3 huruf
  text = re.sub(r'\b\w{1,2}\b', '  ', text)
  #mengahapus tanda baca
  text = text.translate(str.maketrans('','', string.punctuation))
  #hapus whitespace
  text = text.strip()
  #hapus multiple whitespace menjadi single whitespace
  text = re.sub('\s+', ' ', text)
  #remove single char
  text = re.sub(r"\b[a-zA-Z]b", "  ", text)
  #menghapus url
  text = text.replace("http://", "  ").replace("https://", "  ")
  return text

#case folding
def case_folding(text):
  return text.lower()

def bersih(text):
  #Tokenization
  tokens = re.split('\W+', text)
  #Stopword Removal
  Stop = [word for word in tokens if word not in stopword]
  #Stemming
  stemmed_words = [stemmer.stem(w) for w in Stop]

  return ' '.join(stemmed_words)