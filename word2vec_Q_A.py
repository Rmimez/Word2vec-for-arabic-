#!/usr/bin/env python
# coding: utf-8

import pyarabic.araby as araby
import unicodedata as ud
import os
import nltk
import gensim
from gensim import corpora,models,similarities
import re
from sklearn.decomposition import PCA
from matplotlib import pyplot
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt



def punctuation_ar(txt): # this function for token all of the text with deleting punctuation ''.join(c for c in s if not ud.category(c).startswith('P'))
	return ''.join(c for c in txt if not ud.category(c).startswith('P'))

def read_txt(name):
	f=open(name,"r")
	t=f.read()
	t=t.decode("utf8")
	t=araby.strip_tashkeel(t)
	#print t
	return t.split()

#creation du model
def create_model(lst):
	if not(os.path.isfile("/media/rmimez/8A1CED061CECEE5F/etude/soutenance_M2/word2vec/programs/essayer/model.bin")) :
 		corpus=lst
		print type(corpus)
		print corpus
		tok_corp=[nltk.word_tokenize(sent )for sent in corpus]
		#(1->Skip-gram 0->CBOW)
		model=gensim.models.Word2Vec(tok_corp,min_count=1,size=32,sg=0,iter=1000)
		print model
		model.save('model.bin')

def similar_word(model,word):
 # find and print the most similar terms to a word
	try:
		most_similar = model.wv.most_similar( word )
		for term, score in most_similar:
			print (term,score)
		return most_similar
	except Exception as e:
		print "this word not in vocabulary"
 	return None

def vocabular_word():
	try:
  		word_vector = new_model.wv[ word ]
  		print word_vector
 	except Exception as e:
  		print "this word not in vocabulary"
  		raise e
 		 # get a word vector

def graphic(model):
#representation
	X = model[model.wv.vocab]
 	pca = PCA(n_components=2)
 	result = pca.fit_transform(X)
 	# create a scatter plot of the projection
 	pyplot.scatter(result[:, 0], result[:, 1])
 	words = list(model.wv.vocab)
 	for i, word in enumerate(words):
  		pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
 	pyplot.show()

def tsne_plot(model):
    "Creates and TSNE model and plots it"
    labels = []
    tokens = []

    for word in model.wv.vocab:
        tokens.append(model[word])
        labels.append(word)
    
    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
        
    plt.figure(figsize=(16, 16)) 
    for i in range(len(x)):
        plt.scatter(x[i],y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()

path="/media/rmimez/8A1CED061CECEE5F/etude/soutenance_M2/word2vec/datasets/Farasa-master/WikiNewsTruth.txt"
create_model(read_txt(path))

# load model
new_model = models.Word2Vec.load('model.bin')
print(new_model)

word = "ﺪﻋﻭ".decode('utf8', errors='ignore')
#print similar_word(new_model,word)

#print list(new_model.wv.vocab)
for x in list(new_model.wv.vocab):
	print x
"""	
w1="ﺾﻔﺧ"
w2="ﻝﻭﺮﺘﺑ"
w3="ﺔﻳﻭدﻷ"
print new_model.wv.most_similar(positive=[w1,w2], negative=[w3])
"""
#graphic(new_model)
tsne_plot(new_model)

