# %%
import pandas as pd
import numpy as np
import requests
# %%
# Download the data 

docs_url = 'https://github.com/alexeygrigorev/llm-rag-workshop/raw/main/notebooks/documents.json'
docs_response = requests.get(docs_url)
documents_raw = docs_response.json()

documents = []

for course in documents_raw:
    course_name = course['course']

    for doc in course['documents']:
        doc['course'] = course_name
        documents.append(doc)
# %%
documents[2]
# %%
# Want to find 5 most relavant documents for our search query

# convert data to dataframe
df = pd.DataFrame(documents, columns=["course", "section", "question", "text"])
# %%
# note that different courses may have different answers for the same question
df.head()
# %%
df.tail()
# %%
# restrict to a specific course
df[df.course=="data-engineering-zoomcamp"]
# %%
# implement text search
# for that we use vector spaces
# turn document into a vector 
# easiest way to do this: matrix, with each row being a documents
# term-documents matrix: rows: documents, columns: words / tokens 
# sklearn: CountVectorizer
# this representation is called: "bag of words"
# word order does not matter, word order is lost
# sparse matrix

from sklearn.feature_extraction.text import CountVectorizer

# only care about wordss that appear in at least 5 documents: set min_df =5
cv = CountVectorizer(min_df=5)
cv.fit(df.text)
# %%
cv.get_feature_names_out()
# %%
docs_example = [
    "January course details, register now",
    "Course prerequisites listed in January catalog",
    "Submit January course homework by end of month",
    "Register for January course, no prerequisites",
    "January course setup: Python and Google Cloud"
]
# remove stopwords
cv = CountVectorizer(stop_words="english")
cv.fit(docs_example)
# %%
cv.get_feature_names_out()
# %%
# transform result into matrix
X = cv.transform(docs_example)
# %%
X.todense()
# %%
# create a dataframe
pd.DataFrame(X.todense(), columns=cv.get_feature_names_out()).T
# %%
# for the real documents we want to use
cv = CountVectorizer(stop_words='english', min_df=5)
X = cv.fit_transform(df.text)

#names = cv.get_feature_names_out()
#df_docs = pd.DataFrame(X.toarray(), columns=names)
#df_docs
# %%
# the more often the word is, the less important the word is
# to implement this we use TfidfVectorizer
# with that the words get different weights
# we will use for scoring
from sklearn.feature_extraction.text import TfidfVectorizer

cv = TfidfVectorizer(stop_words='english', min_df=5)
X = cv.fit_transform(df.text)

#names = cv.get_feature_names_out()
#df_docs = pd.DataFrame(X.toarray(), columns=names)
#df_docs.round(2)
# %%
X
# %%
query = "Do I need to know python to sign up for the January course?"
# %%
q = cv.transform([query])
q.toarray()
# %%
# show which entries are non-zero
names = cv.get_feature_names_out()
query_dict = dict(zip(names, q.toarray()[0]))
query_dict
# %%
doc_dict = dict(zip(names, X.toarray()[1]))
doc_dict
# %%
# if the document end the term contain the same words the document is relevant
# multiply weight of word in query with the weight of the same word in doc
# sum over all matching terms 
# this gives a measure of similarity how matching the query is for the document (cosine similarity)
# this is done by applying the don product
X.dot(q.T).todense()


# %%
# in sklearn we can use  cosine_similarity
from sklearn.metrics.pairwise import cosine_similarity

# calculate similarity and flatten matrix into a vector
score = cosine_similarity(X, q).flatten().flatten()
# %%
# indices of highest scores
np.argsort(score)[-5:]
# %%
# search over all fields
fields = ["section", "question", "text"]

matrices = {}
vectorizers = {}
for f in fields:
    cv = TfidfVectorizer(stop_words='english', min_df=5)
    X = cv.fit_transform(df[f])
    matrices[f] = X
    vectorizers[f] = cv



# %%
matrices

# %%
n = len(df)

score = np.zeros(n)
query = "I just discovered the course, is it too late to join?"
for f in fields:
    q = vectorizers[f].transform([query])
    X = matrices[f]
    f_score = cosine_similarity(X, q).flatten()
    score = score + f_score


# %%
idx = np.argsort(score)[-5:]

# %%
# relevant documents
df.iloc[idx]
# %%
# we need to add filtering for the specific course
filters = {
    "course": "data-engineering-zoomcamp"
}

for field, value in filters.items():
    mask = (df[field]==value).astype(int)
    score = score * mask
score
# %%
idx = np.argsort(score)[-5:]
df.iloc[idx]
# %%
# question field is more important than the text field
# we can give more score to question than to text -> elasticsearch

n = len(df)

score = np.zeros(n)
query = "I just discovered the course, is it too late to join?"

boosts = {
    "questions": 3,
    #"text": 0.5
}

for f in fields:
    q = vectorizers[f].transform([query])
    X = matrices[f]
    f_score = cosine_similarity(X, q).flatten()
    boost = boosts.get(f, 1.0)
    score = score + boost*f_score
# %%
for field, value in filters.items():
    mask = (df[field]==value).astype(int)
    score = score * mask
score
# %%
idx = np.argsort(score)[-5:]
df.iloc[idx]
# %%
# Put all together
class TextSearch:

    def __init__(self, text_fields):
        self.text_fields = text_fields
        self.matrices = {}
        self.vectorizers = {}

    def fit(self, records, vectorizer_params={}):
        self.df = pd.DataFrame(records)

        for f in self.text_fields:
            cv = TfidfVectorizer(**vectorizer_params)
            X = cv.fit_transform(self.df[f])
            self.matrices[f] = X
            self.vectorizers[f] = cv

    def search(self, query, n_results=10, boost={}, filters={}):
        score = np.zeros(len(self.df))

        for f in self.text_fields:
            b = boost.get(f, 1.0)
            q = self.vectorizers[f].transform([query])
            s = cosine_similarity(self.matrices[f], q).flatten()
            score = score + b * s

        for field, value in filters.items():
            mask = (self.df[field] == value).values
            score = score * mask

        idx = np.argsort(-score)[:n_results]
        results = self.df.iloc[idx]
        return results.to_dict(orient='records')
# %%
index = TextSearch(text_fields=['section', 'question', 'text'])
index.fit(documents)
index.search(
    query='I just singned up. Is it too late to join the course?',
    n_results=5,
    boost={'question': 3.0},
    filters={'course': 'data-engineering-zoomcamp'}
)

# %%
# vectorsearch
# useful, when words don't match exactly
# SVD: compresses matrix, while maintaining most relevant information

from sklearn.decomposition import TruncatedSVD 

X = matrices['text']
cv = vectorizers['text']

svd = TruncatedSVD(n_components=16)
X_emb = svd.fit_transform(X)

# %%
# dense representation = embedding
# sinonyms are reduced into the same concepts / grouped into the same representations
X_emb.shape
# %%
# do the same with the query
query = 'I just singned up. Is it too late to join the course?'

Q = cv.transform([query])
Q_emb = svd.transform(Q)


# %%
Q_emb.shape
# %%
# Now, we compute again the cosine similarity
# That is the idea of the search is the same, only the representations of the documents is different
np.dot(X_emb[0], Q_emb[0])
# %%
# across all documents
score = cosine_similarity(X_emb, Q_emb).flatten()
idx = np.argsort(-score)[:10]
list(df.loc[idx].text)
# %%
# Alternative: Non-Negative Matrix Factorization
# Motivation: in SVD negative values may appear which are difficult to interpret
from sklearn.decomposition import NMF
nmf = NMF(n_components=16)
X_emb = nmf.fit_transform(X)
X_emb[0]
# %%
Q = cv.transform([query])
Q_emb = nmf.transform(Q)
Q_emb[0]

# %%
score = cosine_similarity(X_emb, Q_emb).flatten()
idx = np.argsort(-score)[:10]
list(df.loc[idx].text)
# %%
# get embeddings with BERT
# capture also word order
import torch
from transformers import BertModel, BertTokenizer

# download model and tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")
model.eval()  # Set the model to evaluation mode if not training
# %%
texts = [
    "Yes, we will keep all the materials after the course finishes.",
    "You can follow the course at your own pace after it finishes"
]
encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
encoded_input
# %%
with torch.no_grad():  # Disable gradient calculation for inference
    outputs = model(**encoded_input)
    hidden_states = outputs.last_hidden_state
# %%
# the hidden_states contains the embeddings
# our example contains 2 documents, for each documents a matrix is created
hidden_states.shape
# %%
# calculate mean over first dimension and use this as embedding
# This gives us 2 embeddings for 2 documents
sentence_embeddings = hidden_states.mean(dim=1)
sentence_embeddings.shape
# %%
sentence_embeddings.numpy()

# note that if use a GPU, first you need to move your tensors to CPU
# sentence_embeddings_cpu = sentence_embeddings.cpu()
# %%
# For this example, BERT is not necessary, but it is very useful for more complex documents
# after getting the embedding we use the same approach as previously
# compute the embeddings for all fields: section, question, text
def make_batches(seq, n):
    result = []
    for i in range(0, len(seq), n):
        batch = seq[i:i+n]
        result.append(batch)
    return result
# %%
from tqdm.auto import tqdm
def compute_embeddings(texts, batch_size=8):
    text_batches = make_batches(texts, 8)
    
    all_embeddings = []
    
    for batch in tqdm(text_batches):
        encoded_input = tokenizer(batch, padding=True, truncation=True, return_tensors='pt')
    
        with torch.no_grad():
            outputs = model(**encoded_input)
            hidden_states = outputs.last_hidden_state
            
            batch_embeddings = hidden_states.mean(dim=1)
            batch_embeddings_np = batch_embeddings.cpu().numpy()
            all_embeddings.append(batch_embeddings_np)
    
    final_embeddings = np.vstack(all_embeddings)
    return final_embeddings
# %%
embeddings = {}
 
# fields = ['section', 'question', 'text']

for f in fields:
    print(f'computing embeddings for {f}...')
    embeddings[f] = compute_embeddings(df[f].tolist())
# %%
