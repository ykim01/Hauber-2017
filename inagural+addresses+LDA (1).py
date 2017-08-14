
# coding: utf-8

# In[48]:


from nltk.tokenize import RegexpTokenizer
#from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import re, pprint
import gensim
import os
from urllib import request


# In[49]:


#creatubg ciroys
doc_set = []
i = 0
for doc in os.listdir('/home/ykim/nltk_data/corpora/inaugural/'):
    if os.path.isfile('/home/ykim/nltk_data/corpora/inaugural/' + doc):
        #print(i)
        #print(doc)
        book = open('/home/ykim/nltk_data/corpora/inaugural/'+ doc, 'r').read()
        doc_set.append(book)
        if i < 55:
            i = i + 1
        else:
            break


# In[50]:


#calling python objects
tokenizer = RegexpTokenizer(r'\w+')
p_stemmer = PorterStemmer()


# In[51]:


en_stop = ['a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', "aren't", 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', "can't", 'cannot', 'could', "couldn't", 'did', "didn't", 'do', 'does', "doesn't", 'doing', "don't", 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', "hadn't", 'has', "hasn't", 'have', "haven't", 'having', 'he', "he'd", "he'll", "he's", 'her', 'here', "here's", 'hers', 'herself', 'him', 'himself', 'his', 'how', "how's", 'i', "i'd", "i'll", "i'm", "i've", 'if', 'in', 'into', 'is', "isn't", 'it', "it's", 'its', 'itself', "let's", 'me', 'more', 'most', "mustn't", 'my', 'myself', 'no', 'nor', 'not', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'ought', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 'same', "shan't", 'she', "she'd", "she'll", "she's", 'should', "shouldn't", 'so', 'some', 'such', 'than', 'that', "that's", 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', "there's", 'these', 'they', "they'd", "they'll", "they're", "they've", 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 'very', 'was', "wasn't", 'we', "we'd", "we'll", "we're", "we've", 'were', "weren't", 'what', "what's", 'when', "when's", 'where', "where's", 'which', 'while', 'who', "who's", 'whom', 'why', "why's", 'with', "won't", 'would', "wouldn't", 'you', "you'd", "you'll", "you're", "you've", 'your', 'yours', 'yourself', 'yourselves',
 'will', 'people', 'nation', 'government', 'us','can','may', 'upon']


# In[52]:


#cleaning corpus
new_corpus = []
for i in doc_set:
    raw = i.lower()
    tokens = tokenizer.tokenize(raw)
    #print(tokens)
    
    stopped_tokens = [i for i in tokens if not i in en_stop]
    stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]   
    print(stemmed_tokens)
    
    new_corpus.append(stemmed_tokens)


# In[53]:


#creating dict of corpus
dictionary = corpora.Dictionary(new_corpus)
print(dictionary.token2id)


# In[54]:


#creating bag of words
bags = [dictionary.doc2bow(doc) for doc in new_corpus]


# In[55]:


#prints what id corresponds to what word
print(dictionary[16])


# In[56]:


#outputs each words assignemt id number in dictionary and how many times it shows up in the document
bags[1]


# In[57]:


#lda model used
ldamodel = gensim.models.ldamodel.LdaModel(bags, num_topics=5, id2word = dictionary, passes=20)


# In[58]:


print(ldamodel.print_topics(num_topics=5, num_words=3))


# In[69]:


#test to see what topics first document contains
doc_1 = ldamodel[bags[0]]


# In[70]:


doc_1


# In[79]:


#target document number
docnum = 0


# In[80]:


target_doc_topics = ldamodel[bags[docnum]]


# In[81]:


#getting target document topics
search_topic_array = []
for topic in target_doc_topics:
    search_topic_array.append(topic[0])


# In[82]:


search_topic_array


# In[83]:


#finding similiar topic documents
result_array = []

counter = 0
for bag in bags:
    doc = ldamodel[bag]
    print(doc)
    for t in doc:    
        if t[0] in search_topic_array:
            result_array.append((counter, t[1], t[0]))
    counter += 1


# In[84]:


result_array


# In[85]:


#sorting them by most similar
result_array.sort(key = lambda x:x[1], reverse = True)


# In[86]:


result_array


# In[ ]:




