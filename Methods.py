
# coding: utf-8

# In[6]:


#latin corpus clean up
def latin_Corpus(corpus,tokenizer,lemmatizer,STOP_LIST):
    new_corpus = []
    for i in corpus:
        tokens = tokenizer.tokenize(i)

        stemmed_tokens = lemmatizer.lemmatize(tokens)
        stopped_tokens = [w for w in stemmed_tokens if not w in STOP_LIST]

        new_corpus.append(stopped_tokens)


# In[7]:


#english corpus clean up
def latin_Corpus(corpus,tokenizer,p_stemmer,STOP_LIST):
    new_corpus = []
    for i in corpus:
        raw = i.lower()
        tokens = tokenizer.tokenize(raw)

        stopped_tokens = [i for i in tokens if not i in STOP_LIST]
        stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]   

        new_corpus.append(stemmed_tokens)


# In[3]:


#lda model in use
def ldamodel(bag, num_topics, dictionary, passes, num_words):
    ldamodel = gensim.models.ldamodel.LdaModel(bags, num_topics=20, id2word = dictionary, passes=40)
    print(ldamodel.print_topics(num_topics=20, num_words=3))


# In[1]:


def highest_topic_search(doc_topics):
    result = max(doc_topics, key=lambda item: item[1])
    return result


# In[2]:


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


# In[ ]:




