# %%
import json

with open('documents-with-ids.json', 'rt') as f_in:
    documents = json.load(f_in)
# %%
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
# example usage
elastic_search(
    query="I just discovered the course. Can I still join?",
    course="data-engineering-zoomcamp"
)
# %%
# loop over all queries in the ground truth data and use the function above
import pandas as pd
# %%
df_ground_truth = pd.read_csv('ground-truth-data.csv')
# %%
ground_truth = df_ground_truth.to_dict(orient='records')
# %%
relevance_total = []

# check if doc_id is in the results
for q in tqdm(ground_truth):
    doc_id = q['document']
    results = elastic_search(query=q['question'], course=q['course'])
    relevance = [d['id'] == doc_id for d in results]
    relevance_total.append(relevance)
# %%
# relevance shows if the document id was in the retrieved results from elasticsearch
example = [
    [True, False, False, False, False], # 1, 
    [False, False, False, False, False], # 0
    [False, False, False, False, False], # 0 
    [False, False, False, False, False], # 0
    [False, False, False, False, False], # 0 
    [True, False, False, False, False], # 1
    [True, False, False, False, False], # 1
    [True, False, False, False, False], # 1
    [True, False, False, False, False], # 1
    [True, False, False, False, False], # 1 
    [False, False, True, False, False],  # 1/3
    [False, False, False, False, False], # 0
]

# 1 => 1
# 2 => 1 / 2 = 0.5
# 3 => 1 / 3 = 0.3333
# 4 => 0.25
# 5 => 0.2
# rank => 1 / rank
# none => 0
# %%
# hit rate is 1, if at least one True is in the row
def hit_rate(relevance_total):
    cnt = 0

    for line in relevance_total:
        if True in line:
            cnt = cnt + 1

    return cnt / len(relevance_total)
# %%
# additional takes the position of the result into account
def mrr(relevance_total):
    total_score = 0.0

    for line in relevance_total:
        for rank in range(len(line)):
            if line[rank] == True:
                total_score = total_score + 1 / (rank + 1)

    return total_score / len(relevance_total)
# %%
hit_rate(example)
# %%
mrr(example)
# %%
# hit-rate (recall)
# Mean Reciprocal Rank (mrr)
# %%
# calculate for entire dataset
hit_rate(relevance_total), mrr(relevance_total)
# %%
# repeat the same for minsearch
!wget https://raw.githubusercontent.com/alexeygrigorev/minsearch/main/minsearch.py
#%%
import minsearch

index = minsearch.Index(
    text_fields=["question", "text", "section"],
    keyword_fields=["course", "id"]
)

index.fit(documents)
# %%
def minsearch_search(query, course):
    boost = {'question': 3.0, 'section': 0.5}

    results = index.search(
        query=query,
        filter_dict={'course': course},
        boost_dict=boost,
        num_results=5
    )

    return results
# %%
relevance_total = []

for q in tqdm(ground_truth):
    doc_id = q['document']
    results = minsearch_search(query=q['question'], course=q['course'])
    relevance = [d['id'] == doc_id for d in results]
    relevance_total.append(relevance)
# %%
hit_rate(relevance_total), mrr(relevance_total)
# %%
# Compare with elasticsearch results: minsearch achieves higher metrics
# %%
# create function for evaluation
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
evaluate(ground_truth, lambda q: elastic_search(q['question'], q['course']))
# %%
evaluate(ground_truth, lambda q: minsearch_search(q['question'], q['course']))
# %%
# now we can experiment with thw search parameters, like "boost" and compare results for different parameters
# these experiments are ideally tracked with e.g. MLflow