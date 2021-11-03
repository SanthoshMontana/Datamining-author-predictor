#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import nltk
import pandas as pd
nltk.download('punkt')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import BernoulliNB
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer 
from sklearn.feature_extraction.text import TfidfVectorizer
import string
import warnings
warnings.filterwarnings('ignore')

pre_list = ['Alcott-Lousia May-Eight Cousins.txt',"Alcott-Lousia May-Jo's Boys.txt",'Alcott-Lousia May-Little Men.txt',
                      'Austen-Jane-Mansfield Park.txt','Austen-Jane-Northanger Abbey.txt','Austen-Jane-Persuasion.txt',
                     'Austen-Jane-Pride and Prejudice.txt','Bronte-Charlotte-Jane Eyre.txt','Bronte-Charlotte-Professor.txt',
                     'Bronte-Charlotte-Shirley.txt','Bronte-Charlotte-Villette.txt','Conrad-Joseph-Lord Jim.txt',
                     'Conrad-Joseph-Nostromo.txt','Conrad-Joseph-Secret Agent.txt','Conrad-Joseph-Secret Sharer.txt',
                     'Dickens-Charles-Bleak House.txt','Dickens-Charles-Christmas Carol.txt','Dickens-Charles-Hard Times.txt',
                     'Dickens-Charles-Life And Adventures Of Nicholas Nickleby.txt','Dickens-Charles-Pickwick Papers.txt']

post_list = ['Alcott-Lousia May-Little Women.txt','Austen-Jane-Sense and Sensibility.txt', 'Bronte-Charlotte-Villette.txt',
'Conrad-Joseph-Heart of Darkness.txt',
'Dickens-Charles-David Copperfield.txt']

def process_books(list_for_run, row_type):
    """Skeleton for processing multiple files, yielding a dataframe.
    """
    book_file_list = list_for_run
     
    results = [] 
    for bname in book_file_list:
        book_str = read_book(bname)
        temp_df = process(book_str, bname, row_type)
        results.append(temp_df)
    final_df = pd.concat(results, ignore_index = True)
    return final_df

def read_book(filename):
    file = open(filename,"r")
    return file.read()
    
def process(book, author, row_type):
    """book is the book's text as a string. It should be broken into rows 
    of n tokens. 'author' is the file name, from which the author is extracted.
    """
    if(row_type == 0):
        rows = get_rows(book)
        df = pd.DataFrame(rows, columns=["text"])
        author_splice = author.split('-')[0:2]
        author_splice = ' '.join(author_splice)
        df['author'] = author_splice
        return df
    elif(row_type == 1):
        rows = get_rows_stemmer(book)
        df = pd.DataFrame(rows, columns=["text"])
        author_splice = author.split('-')[0:2]
        author_splice = ' '.join(author_splice)
        df['author'] = author_splice
        return df

def get_rows(text, chunk_size = 500):
        """Chunk book text into rows of n tokens.
        Here, we instead just split into lines. """
        toks = word_tokenize(text)
        to_pad = chunk_size - len(toks) % chunk_size
        toks.extend(['' for i in range(to_pad)])
        array = np.array(toks)
        array = array.reshape(-1, chunk_size)
        return [' '.join(e) for e in array]
        
    
    
    
def get_rows_stemmer(text, chunk_size = 500):
        """Chunk book text into rows of n tokens.
        Here, we instead just split into lines. """
        ps = PorterStemmer()
        toks = word_tokenize(text)
        to_pad = chunk_size - len(toks) % chunk_size
        toks.extend(['' for i in range(to_pad)])
        toks =  [ps.stem(token) for token in toks]
        array = np.array(toks)
        array = array.reshape(-1, chunk_size)
        return [' '.join(e) for e in array]    
    
def processing_v2(input_list):
    output_list = []
    for strings in input_list:
        output_list.append(strings.translate(str.maketrans('', '', string.punctuation)))
    return output_list





#normal processing 
print("type process_books() to test.")
result = process_books(pre_list,0)
corpus = result['text']
author_list = result['author']
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(corpus)
clf = MultinomialNB().fit(X_train_counts, author_list)
new_result = process_books(post_list,0)
new_data = new_result['text']
new_data_authors = new_result['author']
X_new_counts = count_vect.transform(new_data)
predicted = clf.predict(X_new_counts)
print("Multinomial: (regular pre-process) " + str(accuracy_score(new_data_authors,predicted)))
cm = confusion_matrix(new_data_authors, predicted)
df_show = pd.DataFrame()
df_show = pd.DataFrame(cm, index = ["Alcott","Austen","Bronte", "Conrad", "Dickens"], columns= ["Alcott","Austen","Bronte", "Conrad", "Dickens"])
print(df_show)

clf = BernoulliNB().fit(X_train_counts, author_list)
predicted = clf.predict(X_new_counts)
print()
print("Bernoulli: (regular pre-process) " + str(accuracy_score(new_data_authors,predicted)))
cm = confusion_matrix(new_data_authors, predicted)
df_show = pd.DataFrame()
df_show = pd.DataFrame(cm, index = ["Alcott","Austen","Bronte", "Conrad", "Dickens"], columns= ["Alcott","Austen","Bronte", "Conrad", "Dickens"])
print(df_show)


#stop words and tfdif
stop_words = set(stopwords.words('english')) 
my_vectorizer = TfidfVectorizer(stop_words)
X_train_counts = my_vectorizer.fit_transform(corpus)
clf = MultinomialNB().fit(X_train_counts, author_list)
X_new_counts = my_vectorizer.transform(processing_v2(new_data))
predicted = clf.predict(X_new_counts)
print("Multinomial: (Tifdif vectorizer and words pre-process) " + str(accuracy_score(new_data_authors,predicted)))
cm = confusion_matrix(new_data_authors, predicted)
df_show = pd.DataFrame()
df_show = pd.DataFrame(cm, index = ["Alcott","Austen","Bronte", "Conrad", "Dickens"], columns= ["Alcott","Austen","Bronte", "Conrad", "Dickens"])
print(df_show)

print()
X_train_counts = my_vectorizer.fit_transform(corpus)
clf = BernoulliNB().fit(X_train_counts, author_list)
predicted = clf.predict(X_new_counts)
print("Bernoulli: (Tifdif vectorize and stop words pre-process) " + str(accuracy_score(new_data_authors,predicted)))
cm = confusion_matrix(new_data_authors, predicted)
df_show = pd.DataFrame()
df_show = pd.DataFrame(cm, index = ["Alcott","Austen","Bronte", "Conrad", "Dickens"], columns= ["Alcott","Austen","Bronte", "Conrad", "Dickens"])
print(df_show)


#punctuation processing 
X_train_counts = count_vect.fit_transform(processing_v2(corpus))
clf = MultinomialNB().fit(X_train_counts, author_list)
X_new_counts = count_vect.transform(processing_v2(new_data))
predicted = clf.predict(X_new_counts)
print()
print("Multinomial: (punctuatuion pre-process) "+ str(accuracy_score(new_data_authors,predicted)))
cm = confusion_matrix(new_data_authors, predicted)
df_show = pd.DataFrame()
df_show = pd.DataFrame(cm, index = ["Alcott","Austen","Bronte", "Conrad", "Dickens"], columns= ["Alcott","Austen","Bronte", "Conrad", "Dickens"])
print(df_show)


clf = BernoulliNB().fit(X_train_counts, author_list)
predicted = clf.predict(X_new_counts)
print()
print("Bernoulli: (punctuatuion pre-process) "+ str(accuracy_score(new_data_authors,predicted)))
cm = confusion_matrix(new_data_authors, predicted)
df_show = pd.DataFrame()
df_show = pd.DataFrame(cm, index = ["Alcott","Austen","Bronte", "Conrad", "Dickens"], columns= ["Alcott","Austen","Bronte", "Conrad", "Dickens"])
print(df_show)

#stemming starts here. apologies for the wait. (had to reprocess the data):
print()
result = process_books(pre_list,1)
corpus = result['text']
author_list = result['author']
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(corpus)
clf = MultinomialNB().fit(X_train_counts, author_list)
new_result = process_books(post_list,1)
new_data = new_result['text']
new_data_authors = new_result['author']
X_new_counts = count_vect.transform(new_data)
predicted = clf.predict(X_new_counts)
print("Multinomial: (stem pre-process) " + str(accuracy_score(new_data_authors,predicted)))
cm = confusion_matrix(new_data_authors, predicted)
df_show = pd.DataFrame()
df_show = pd.DataFrame(cm, index = ["Alcott","Austen","Bronte", "Conrad", "Dickens"], columns= ["Alcott","Austen","Bronte", "Conrad", "Dickens"])
print(df_show)
print()

clf = BernoulliNB().fit(X_train_counts, author_list)
predicted = clf.predict(X_new_counts)
print("Bernoulli: (stem pre-process) " + str(accuracy_score(new_data_authors,predicted)))
cm = confusion_matrix(new_data_authors, predicted)
df_show = pd.DataFrame()
df_show = pd.DataFrame(cm, index = ["Alcott","Austen","Bronte", "Conrad", "Dickens"], columns= ["Alcott","Austen","Bronte", "Conrad", "Dickens"])
print(df_show)
print()


# In[ ]:





# In[ ]:





# In[ ]:




