# %%
from sentence_transformers import SentenceTransformer
# %%
# model to create embeddings
# sbert.net/docs/sentence_transformer/pretrained_model.html
model_name = 'multi-qa-distilbert-cos-v1'
model = SentenceTransformer(model_name)
# %%
# Q1
user_question = "I just discovered the course. Can I still join it?"
v = model.encode(user_question)
# %%
v.shape
# %%
# Prepare the documents
import requests 

base_url = 'https://github.com/DataTalksClub/llm-zoomcamp/blob/main'
relative_url = '03-vector-search/eval/documents-with-ids.json'
docs_url = f'{base_url}/{relative_url}?raw=1'
docs_response = requests.get(docs_url)
documents = docs_response.json()
print(len(documents))
# %%
# Q2
#Create a list `embeddings`
#Iterate over each document
#`qa_text = f'{question} {text}'`
#compute the embedding for `qa_text`, append to `embeddings`
#At the end, let `X = np.array(embeddings)` (import numpy as np)
from tqdm.auto import tqdm
import numpy as np
embeddings = []
for doc in tqdm(documents):
    question = doc["question"]
    text = doc["text"]
    qa_text = f'{question} {text}'
    emb = model.encode(qa_text)
    embeddings.append(emb)
X = np.array(embeddings)
print(X.shape)
# %%
# Q3
# cosine similarities
scores = X.dot(v)
print(max(scores))
# %%
# Q3
class VectorSearchEngine():
    def __init__(self, documents, embeddings):
        self.documents = documents
        self.embeddings = embeddings

    def search(self, v_query, num_results=10):
        scores = self.embeddings.dot(v_query)
        idx = np.argsort(-scores)[:num_results]
        return [self.documents[i] for i in idx]

search_engine = VectorSearchEngine(documents=documents, embeddings=X)
search_engine.search(v, num_results=5)
# %%
# Q4
import pandas as pd

base_url = 'https://github.com/DataTalksClub/llm-zoomcamp/blob/main'
relative_url = '03-vector-search/eval/ground-truth-data.csv'
ground_truth_url = f'{base_url}/{relative_url}?raw=1'

df_ground_truth = pd.read_csv(ground_truth_url)
df_ground_truth = df_ground_truth[df_ground_truth.course == 'machine-learning-zoomcamp']
ground_truth = df_ground_truth.to_dict(orient='records')
print(len(df_ground_truth))
# %%
df_ground_truth.head()
# %%
ground_truth[:3]
# %%
relevance_total = []

# check if doc_id is in the results
for q in tqdm(ground_truth):
    doc_id = q['document']
    v = model.encode(q["question"])
    results = search_engine.search(v, num_results=5)
    relevance = [d['id'] == doc_id for d in results]
    relevance_total.append(relevance)
# %%
relevance_total

# %%
def hit_rate(relevance_total):
    cnt = 0

    for line in relevance_total:
        if True in line:
            cnt = cnt + 1

    return cnt / len(relevance_total)
# %%
print(hit_rate(relevance_total))

# %%
# Q5
# index documents with elasticseach
from elasticsearch import Elasticsearch

es_client = Elasticsearch('http://localhost:9200') 

index_settings = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0
    },
    "mappings": {
        "properties": {
            "text": {"type": "text"},
            "section": {"type": "text"},
            "question": {"type": "text"},
            "course": {"type": "keyword"},
            "id": {"type": "keyword"},
            
            "question_text_vector": {
                "type": "dense_vector",
                "dims": 384,
                "index": True,
                "similarity": "cosine"
            },
        }
    }
}
index_name = "course-questions"

es_client.indices.delete(index=index_name, ignore_unavailable=True)
es_client.indices.create(index=index_name, body=index_settings)
# %%
from tqdm.auto import tqdm

for doc in tqdm(documents):
    es_client.index(index=index_name, document=doc)
# %%
# define search query
def elastic_search(query, course):
    search_query = {
        "size": 5,
        "query": {
            "bool": {
                "must": {
                    "multi_match": {
                        "query": query,
                        "fields": ["question^3", "text", "section"],
                        "type": "best_fields"
                    }
                },
                "filter": {
                    "term": {
                        "course": course
                    }
                }
            }
        }
    }

    response = es_client.search(index=index_name, body=search_query)
    
    result_docs = []
    
    for hit in response['hits']['hits']:
        result_docs.append(hit['_source'])
    
    return result_docs
# %%

user_question = "I just discovered the course. Can I still join it?"
v = model.encode(user_question)
elastic_search(
    query=v,
    course="machine-learning-zoomcamp"
)
# %%
# Q6
relevance_total = []
for q in tqdm(ground_truth):
    doc_id = q['document']
    results = elastic_search(query=q['question'], course=q['course'])
    relevance = [d['id'] == doc_id for d in results]
    relevance_total.append(relevance)

# %%
print(hit_rate(relevance_total))
# %%
