
# coding: utf-8


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


#for a regular text file, file opening and reading
#file1 = '/home/ykim/Desktop/wodeham-b1-d3-qun-clean-html.xml'
#book1 = open(file1, 'r').read()

#for a xml doc
tree = etree.parse("/home/ykim/Desktop/wodeham-b1-d3-qun-clean-html.xml")
doc_set = tree.xpath("//p//text()")



doc_set


tokenizer = RegexpTokenizer(r'\w+')
lemmatizer = LemmaReplacer('latin')


corpus = doc_set


#how many paragraphs there are
len(corpus)



STOPS_LIST = ['ab', 'ac', 'ad', 'adhic', 'aliqui', 'aliquis', 'an', 'ante', 'apud', 'at', 'atque', 'aut', 'autem', 'cum', 'cur', 'de', 'deinde', 'dum', 'ego', 'enim', 'ergo', 'es', 'est', 'et', 'etiam', 'etsi', 'ex', 'fio', 'haud', 'hic', 'iam', 'idem', 'igitur', 'ille', 'in', 'infra', 'inter', 'interim', 'ipse', 'is', 'ita', 'magis', 'modo', 'mox', 'nam', 'ne', 'nec', 'necque', 'neque', 'nisi', 'non', 'nos', 'o', 'ob', 'per', 'possum', 'post', 'pro', 'quae', 'quam', 'quare', 'qui', 'quia', 'quicumque', 'quidem', 'quilibet', 'quis', 'quisnam', 'quisquam', 'quisque', 'quisquis', 'quo', 'quoniam', 'sed', 'si', 'sic', 'sive', 'sub', 'sui', 'sum', 'super', 'suus', 'tam', 'tamen', 'trans', 'tu', 'tum', 'ubi', 'uel', 'uero', 'unus', 'ut', 'sum1', 'qui1', 'edo1', 'quis1', 'meus', 'tantus', 'sum1', 'suum', 'quantus', 'quidam', 'eo1', "dico1", 'dico2', 'f', 'quasi', 'neo1', 'inquam', 'vel', 'que', "suo"]


#corpus clean up
latin_Corpus(corpus,tokenizer,lemmatizer,STOP_LIST)


#creating dict of campus
dictionary = corpora.Dictionary(new_corpus)


# In[175]:


#removes words that frequent more than 500 times
dictionary.filter_n_most_frequent(500)


# In[176]:


#print(dictionary.token2id)


# In[177]:


#creates bag of words
bags = [dictionary.doc2bow(doc) for doc in new_corpus]


# In[199]:


#lda model in use
ldamodel = gensim.models.ldamodel.LdaModel(bags, num_topics=20, id2word = dictionary, passes=40)


# In[200]:


print(ldamodel.print_topics(num_topics=20, num_words=3))


# In[238]:


#example of the topics a paragraph has
doc_1 = ldamodel[bags[1]]


# In[239]:


doc_1


# In[240]:


def highest_topic_search(doc_topics):
    result = max(doc_topics, key=lambda item: item[1])
    return result


# In[241]:


target_topic = highest_topic_search(doc_1)


# In[242]:


#finds the paragraph with the same topic and highest %
def find_other_highest(target_t, bags, percentage):
    result_array = []
    counter = 0
    for bag in bags:
        doc = ldamodel[bag]
        
        for t in doc:    
            if (t[0] == target_t[0]) and (t[1] >= percentage):
                result_array.append((counter, t[1], t[0]))
        counter += 1
    result_array.sort(key = lambda x:x[1], reverse = True)   
    return result_array    


# In[244]:


find_other_highest(target_topic, bags, 0.50)


# In[152]:


#target paragraph
paranum = 0


# In[153]:


#get target paragraph's topics
target_doc_topics = ldamodel[bags[paranum]]


# In[154]:


search_topic_array = []
for topic in target_doc_topics:
    search_topic_array.append(topic[0])


# In[155]:


search_topic_array


# In[156]:


#find the related topics
result_array = []
counter = 0
for bag in bags:
    doc = ldamodel[bag]
    print(doc)
    for t in doc:    
        if t[0] in search_topic_array:
            result_array.append((counter, t[1], t[0]))
    counter += 1


# In[160]:


#result_array


# In[158]:


#sort them by most similar
result_array.sort(key = lambda x:x[1], reverse = True)


# In[159]:


result_array


# In[ ]:




