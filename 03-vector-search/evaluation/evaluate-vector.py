# %%
import json

with open('documents-with-ids.json', 'rt') as f_in:
    documents = json.load(f_in)
# %%
from sentence_transformers import SentenceTransformer
# %%
# model to create embeddings
# sbert.net/docs/sentence_transformer/pretrained_model.html
model_name = 'multi-qa-MiniLM-L6-cos-v1'
model = SentenceTransformer(model_name)
# %%
# test model on an example
v = model.encode('I just discovered the course. Can I still join?')
# %%
len(v)
# %%
# create embeddings for questions
# create embeddings for answers
# create embedding for combination of question and answer
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
            "question_vector": {
                "type": "dense_vector",
                "dims": 384,
                "index": True,
                "similarity": "cosine"
            },
            "text_vector": {
                "type": "dense_vector",
                "dims": 384,
                "index": True,
                "similarity": "cosine"
            },
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
# %%
# loop over documents
for doc in tqdm(documents):
    question = doc['question']
    text = doc['text']
    qt = question + ' ' + text

    doc['question_vector'] = model.encode(question)
    doc['text_vector'] = model.encode(text)
    doc['question_text_vector'] = model.encode(qt)
# %%
for doc in tqdm(documents):
    es_client.index(index=index_name, document=doc)
# %%
# example query
query = 'I just discovered the course. Can I still join it?'
# %%
# encode query
v_q = model.encode(query)
# %%
# create search query
def elastic_search_knn(field, vector, course):
    knn = {
        "field": field,
        "query_vector": vector,
        "k": 5,
        "num_candidates": 10000,
        "filter": {
            "term": {
                "course": course
            }
        }
    }

    search_query = {
        "knn": knn,
        "_source": ["text", "section", "question", "course", "id"]
    }

    es_results = es_client.search(
        index=index_name,
        body=search_query
    )
    
    result_docs = []
    
    for hit in es_results['hits']['hits']:
        result_docs.append(hit['_source'])

    return result_docs
# %%
elastic_search_knn("question_vector", v_q, "data-engineering-zoomcamp")
# %%
def question_vector_knn(q):
    question = q['question']
    course = q['course']

    v_q = model.encode(question)

    return elastic_search_knn('question_vector', v_q, course)
# %%
import pandas as pd
# %%
df_ground_truth = pd.read_csv('ground-truth-data.csv')
# %%
ground_truth = df_ground_truth.to_dict(orient='records')
# %%
ground_truth[0]
# %%
def hit_rate(relevance_total):
    cnt = 0

    for line in relevance_total:
        if True in line:
            cnt = cnt + 1

    return cnt / len(relevance_total)
# %%
def mrr(relevance_total):
    total_score = 0.0

    for line in relevance_total:
        for rank in range(len(line)):
            if line[rank] == True:
                total_score = total_score + 1 / (rank + 1)

    return total_score / len(relevance_total)
# %%
def evaluate(ground_truth, search_function):
    relevance_total = []

    for q in tqdm(ground_truth):
        doc_id = q['document']
        results = search_function(q)
        relevance = [d['id'] == doc_id for d in results]
        relevance_total.append(relevance)

    return {
        'hit_rate': hit_rate(relevance_total),
        'mrr': mrr(relevance_total),
    }
# %%
evaluate(ground_truth, question_vector_knn)
# %%
# Compare to elasticsearch text only: the results of the vector search show higher metrics
# But elasticsearch with only text was faster
# %%
# evaluate on "text" (=answer)
def text_vector_knn(q):
    question = q['question']
    course = q['course']

    v_q = model.encode(question)

    return elastic_search_knn('text_vector', v_q, course)

# %%
evaluate(ground_truth, text_vector_knn)
# %%
# metrics are higher using the text field for evaluation
# 
# %%
def question_text_vector_knn(q):
    question = q['question']
    course = q['course']

    v_q = model.encode(question)

    return elastic_search_knn('question_text_vector', v_q, course)

evaluate(ground_truth, question_text_vector_knn)
# %%
# evaluation metrics are higher using the combination of question and answer for evaluation
# %%
# query all 3 dense vecors with one vector
def elastic_search_knn_combined(vector, course):
    search_query = {
        "size": 5,
        "query": {
            "bool": {
                "must": [
                    {
                        "script_score": {
                            "query": {
                                "term": {
                                    "course": course
                                }
                            },
                            "script": {
                                "source": """
                                    cosineSimilarity(params.query_vector, 'question_vector') + 
                                    cosineSimilarity(params.query_vector, 'text_vector') + 
                                    cosineSimilarity(params.query_vector, 'question_text_vector') + 
                                    1
                                """,
                                "params": {
                                    "query_vector": vector
                                }
                            }
                        }
                    }
                ],
                "filter": {
                    "term": {
                        "course": course
                    }
                }
            }
        },
        "_source": ["text", "section", "question", "course", "id"]
    }

    es_results = es_client.search(
        index=index_name,
        body=search_query
    )
    
    result_docs = []
    
    for hit in es_results['hits']['hits']:
        result_docs.append(hit['_source'])

    return result_docs

# %%
def vector_combined_knn(q):
    question = q['question']
    course = q['course']

    v_q = model.encode(question)

    return elastic_search_knn_combined(v_q, course)

evaluate(ground_truth, vector_combined_knn)
# %%