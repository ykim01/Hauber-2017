
# coding: utf-8

# In[1]:


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


#file1 = '/home/ykim/Desktop/wodeham-b1-d3-qun-clean-html.xml'
#book1 = open(file1, 'r').read()


# In[3]:


tree = etree.parse("/home/ykim/Desktop/wodeham-b1-d3-qun-clean-html.xml")
corpus = tree.xpath("//p//text()")


# In[4]:


#doc_set


# In[5]:


tokenizer = RegexpTokenizer(r'\w+')
lemmatizer = LemmaReplacer('latin')


# In[6]:


#how many paragraphs there are
len(corpus)


# In[7]:


STOP_LIST = ['ab', 'ac', 'ad', 'adhic', 'aliqui', 'aliquis', 'an', 'ante', 'apud', 'at', 'atque', 'aut', 'autem', 'cum', 'cur', 'de', 'deinde', 'dum', 'ego', 'enim', 'ergo', 'es', 'est', 'et', 'etiam', 'etsi', 'ex', 'fio', 'haud', 'hic', 'iam', 'idem', 'igitur', 'ille', 'in', 'infra', 'inter', 'interim', 'ipse', 'is', 'ita', 'magis', 'modo', 'mox', 'nam', 'ne', 'nec', 'necque', 'neque', 'nisi', 'non', 'nos', 'o', 'ob', 'per', 'possum', 'post', 'pro', 'quae', 'quam', 'quare', 'qui', 'quia', 'quicumque', 'quidem', 'quilibet', 'quis', 'quisnam', 'quisquam', 'quisque', 'quisquis', 'quo', 'quoniam', 'sed', 'si', 'sic', 'sive', 'sub', 'sui', 'sum', 'super', 'suus', 'tam', 'tamen', 'trans', 'tu', 'tum', 'ubi', 'uel', 'uero', 'unus', 'ut', 'sum1', 'qui1', 'edo1', 'quis1', 'meus', 'tantus', 'sum1', 'suum', 'quantus', 'quidam', 'eo1', "dico1", 'dico2', 'f', 'quasi', 'neo1', 'inquam', 'vel', 'que', "suo"]


# In[8]:


#corpus clean up
new_corpus = []
for i in corpus:
    tokens = tokenizer.tokenize(i)
    #print(tokens)
    
    stemmed_tokens = lemmatizer.lemmatize(tokens)
    stopped_tokens = [w for w in stemmed_tokens if not w in STOP_LIST]
    
    new_corpus.append(stopped_tokens)
    #print(stemmed_tokens)


# In[29]:


new_corpus


# In[9]:


#creating dict of campus
dictionary = corpora.Dictionary(new_corpus)


# In[10]:


#removes words that frequent more than 500 times
dictionary.filter_n_most_frequent(500)


# In[11]:


#print(dictionary.token2id)


# In[12]:


#creates bag of words
bags = [dictionary.doc2bow(doc) for doc in new_corpus]


# In[13]:


#lda model in use
ldamodel = gensim.models.ldamodel.LdaModel(bags, num_topics=20, id2word = dictionary, passes=40)


# In[14]:


print(ldamodel.print_topics(num_topics=20, num_words=3))


# In[15]:


#example of the topics a paragraph has
doc_1 = ldamodel[bags[1]]


# In[16]:


doc_1


# In[17]:


def highest_topic_search(doc_topics):
    result = max(doc_topics, key=lambda item: item[1])
    return result


# In[18]:


target_topic = highest_topic_search(doc_1)


# In[19]:


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


# In[20]:


find_other_highest(target_topic, bags, 0.50)


# In[21]:


#target paragraph
paranum = 0


# In[22]:


#get target paragraph's topics
target_doc_topics = ldamodel[bags[paranum]]


# In[23]:


search_topic_array = []
for topic in target_doc_topics:
    search_topic_array.append(topic[0])


# In[24]:


search_topic_array


# In[25]:


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


# In[26]:


#result_array


# In[27]:


#sort them by most similar
result_array.sort(key = lambda x:x[1], reverse = True)


# In[28]:


result_array


# In[ ]:




