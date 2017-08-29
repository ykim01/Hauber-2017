
# coding: utf-8

# In[1]:


from Methods import latin_Corpus, ldamodel, highest_topic_search, find_other_highest, related_topics
from cltk.stem.lemma import LemmaReplacer
from nltk.tokenize import RegexpTokenizer
from cltk.stop.latin.stops import STOPS_LIST
from cltk.vector.word2vec import get_sims
from gensim import corpora, models
import re, pprint
from urllib import request
import gensim
from cltk.corpus.utils.importer import CorpusImporter
corpus_importer = CorpusImporter('latin')
corpus_importer.import_corpus('latin_models_cltk')
import os
from lxml import etree


# In[2]:


#for a regular text file, file opening and reading
#file1 = '/home/ykim/Desktop/wodeham-b1-d3-qun-clean-html.xml'
#book1 = open(file1, 'r').read()

#for a xml doc
tree = etree.parse("/home/ykim/Desktop/wodeham-b1-d3-qun-clean-html.xml")
corpus = tree.xpath("//p//text()")

#corpus


# In[3]:


tokenizer = RegexpTokenizer(r'\w+')
lemmatizer = LemmaReplacer('latin')


# In[4]:


#how many paragraphs there are
len(corpus)


# In[5]:


STOP_LIST = ['ab', 'ac', 'ad', 'adhic', 'aliqui', 'aliquis', 'an', 'ante', 'apud', 'at', 'atque', 'aut', 'autem', 'cum', 'cur', 'de', 'deinde', 'dum', 'ego', 'enim', 'ergo', 'es', 'est', 'et', 'etiam', 'etsi', 'ex', 'fio', 'haud', 'hic', 'iam', 'idem', 'igitur', 'ille', 'in', 'infra', 'inter', 'interim', 'ipse', 'is', 'ita', 'magis', 'modo', 'mox', 'nam', 'ne', 'nec', 'necque', 'neque', 'nisi', 'non', 'nos', 'o', 'ob', 'per', 'possum', 'post', 'pro', 'quae', 'quam', 'quare', 'qui', 'quia', 'quicumque', 'quidem', 'quilibet', 'quis', 'quisnam', 'quisquam', 'quisque', 'quisquis', 'quo', 'quoniam', 'sed', 'si', 'sic', 'sive', 'sub', 'sui', 'sum', 'super', 'suus', 'tam', 'tamen', 'trans', 'tu', 'tum', 'ubi', 'uel', 'uero', 'unus', 'ut', 'sum1', 'qui1', 'edo1', 'quis1', 'meus', 'tantus', 'sum1', 'suum', 'quantus', 'quidam', 'eo1', "dico1", 'dico2', 'f', 'quasi', 'neo1', 'inquam', 'vel', 'que', "suo"]


# In[19]:


#corpus clean up
new_corpus = latin_Corpus(corpus, tokenizer,lemmatizer,STOP_LIST)

#new_corpus


# In[7]:


#creating dict of campus
dictionary = corpora.Dictionary(new_corpus)

#removes words that frequent more than 500 times
dictionary.filter_n_most_frequent(500)


# In[8]:


#print(dictionary.token2id)


# In[9]:


#creates bag of words
bags = [dictionary.doc2bow(doc) for doc in new_corpus]


# In[10]:


#lda model in use
ldamodel = ldamodel(gensim, bags, 20, dictionary, 40,3)


# In[11]:


#example of the topics a paragraph has
doc_1 = ldamodel[bags[1]]

doc_1


# In[12]:


target_topic = highest_topic_search(doc_1)


# In[13]:


find_other_highest(target_topic, bags, 0.50,ldamodel)


# In[14]:


#target paragraph
paranum = 0

#get target paragraph's topics
target_doc_topics = ldamodel[bags[paranum]]


# In[15]:


search_topic_array = []
for topic in target_doc_topics:
    search_topic_array.append(topic[0])

search_topic_array


# In[16]:


#find the related topics
result_array = related_topics(search_topic_array, bags, ldamodel)

#result_array


# In[17]:


#sort them by most similar
result_array.sort(key = lambda x:x[1], reverse = True)

result_array


# In[ ]:




