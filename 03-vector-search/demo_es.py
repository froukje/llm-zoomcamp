# %%
#1. Create documents (from documents.json)
#2. Use a pretrained model (from Hugging Face) to get the embeddings of the documents
#3. Index embeddings into elastic search database

#- A document is a collection of fields with their associated values.
#- To work with ElasticSearch you have to organize your data into documents, and then add all your documents to an index.
#* index is a collection of documents that is stored in a highliy optimized format designed to perform efficient searches.
# %%
## Step 1: Prepare documents
# %%
import json

with open('documents.json', 'rt') as f_in:
    docs_raw = json.load(f_in)
# %%
documents = []

for course_dict in docs_raw:
    for doc in course_dict['documents']:
        doc['course'] = course_dict['course']
        documents.append(doc)

documents[1]
# %%
## Step 2: Create Embeddings using Pretrained Models

#Sentence Transformers documentation here: https://www.sbert.net/docs/sentence_transformer/pretrained_models.html
# %%
# This is a new library compared to the previous modules. 
# Please perform "pip install sentence_transformers==2.7.0"
from sentence_transformers import SentenceTransformer

# if you get an error do the following:
# 1. Uninstall numpy 
# 2. Uninstall torch
# 3. pip install numpy==1.26.4
# 4. pip install torch
# run the above cell, it should work
model = SentenceTransformer("all-mpnet-base-v2")

# %%
len(model.encode("This is a simple sentence"))
# %%
documents[1]
# %%
#created the dense vector using the pre-trained model
operations = []
for doc in documents:
    # Transforming the title into an embedding using the model
    doc["text_vector"] = model.encode(doc["text"]).tolist()
    operations.append(doc)
# %%
## Step 3: Setup ElasticSearch connection
# %%
from elasticsearch import Elasticsearch
es_client = Elasticsearch('http://localhost:9200') 

es_client.info()
# %%
# Step 4: Create Mappings and Index
# * Mapping is the process of defining how a document, and the fields it contains, are stored and indexed.
# * Each document is a collection of fields, which each have their own data type.
# * We can compare mapping to a database schema in how it describes the fields and properties that documents hold, the datatype of each field (e.g., string, integer, or date), and how those fields should be indexed and stored
# %%
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
            "course": {"type": "keyword"} ,
            "text_vector": {"type": "dense_vector", "dims": 768, "index": True, "similarity": "cosine"},
        }
    }
}
# %%
index_name = "course-questions"

es_client.indices.delete(index=index_name, ignore_unavailable=True)
es_client.indices.create(index=index_name, body=index_settings)
# %%
# Step 5: Add documents into index
for doc in operations:
    try:
        es_client.index(index=index_name, document=doc)
    except Exception as e:
        print(e)
# %%
# Step 6: Create end user query
# %%
search_term = "windows or mac?"
vector_search_term = model.encode(search_term)
# %%
query = {
    "field": "text_vector",
    "query_vector": vector_search_term,
    "k": 5,
    "num_candidates": 10000, 
}
# %%
res = es_client.search(index=index_name, knn=query, source=["text", "section", "question", "course"])
res["hits"]["hits"]

# %%
# Step 7: Perform Keyword search with Semantic Search (Hybrid/Advanced Search)

# %%
# Note: I made a minor modification to the query shown in the notebook here
# (compare to the one shown in the video)
# Included "knn" in the search query (to perform a semantic search) along with the filter  
knn_query = {
    "field": "text_vector",
    "query_vector": vector_search_term,
    "k": 5,
    "num_candidates": 10000
}
# %%
response = es_client.search(
    index=index_name,
    query={
        "match": {"course": "data-engineering-zoomcamp"},
    },
    knn=knn_query,
    size=5,
    explain=True
)
# %%
response["hits"]["hits"]
# %%
#References
#https://logz.io/blog/elasticsearch-mapping/#:~:text=Within%20a%20search%20engine%2C%20mapping,indexes%20and%20stores%20its%20fields

#https://www.sbert.net/docs/sentence_transformer/pretrained_models.html

#https://www.elastic.co/search-labs/tutorials

#https://www.elastic.co/search-labs/blog/text-similarity-search-with-vectors-in-elasticsearch