# %%
import requests 

docs_url = 'https://github.com/DataTalksClub/llm-zoomcamp/blob/main/01-intro/documents.json?raw=1'
docs_response = requests.get(docs_url)
documents_raw = docs_response.json()

documents = []

for course in documents_raw:
    course_name = course['course']

    for doc in course['documents']:
        doc['course'] = course_name
        documents.append(doc)
# %%
from elasticsearch import Elasticsearch
# %%
url = "http://localhost:9200"
es_client = Elasticsearch(url)
# %%
es_client.info()
#%%
# create an index

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
            "course": {"type": "keyword"} 
        }
    }
}

index_name = "course-questions"
es_client.indices.create(index=index_name, body=index_settings)
# %%
# index the data
from tqdm.auto import tqdm
for doc in tqdm(documents):
    es_client.index(index=index_name, document=doc)
# %%
query = "How do I execute a command in a running docker container?"

def elastic_search(query, course=None):
    search_query = {
        "size": 3,
        "query": {
                "bool": {
                "must": {
                    "multi_match": {
                        "query": query,
                        "fields": ["question^4", "text", "section"],
                        "type": "best_fields"
                        }
                },
                "filter": {
                    "term": {
                        "course": "machine-learning-zoomcamp"
                    }
                }
            }
        }
    }
    response = es_client.search(index=index_name, body=search_query)

    result_docs = []
    result_scores = []
    for hit in response["hits"]["hits"]:
        result_docs.append(hit["_source"])
        result_scores.append(hit["_score"])

    return result_docs, result_scores
# %%
docs, scores = elastic_search(query)
print(scores)
for doc in docs:
    print(doc["question"]) 
    print(doc["text"]) 
# %%
question = "How do I execute a command in a running docker container?"
# %%

# %%
def build_prompt(results):
    prompt_template = """
        You're a course teaching assistant. Answer the QUESTION based on the CONTEXT from the FAQ database.
        Use only the facts from the CONTEXT when answering the QUESTION.

        QUESTION: {question}

        CONTEXT:
            {context}
        """.strip()
    
    context_template = """
        Q: {question}   
        A: {text}
    """.strip()
    
    context =""

    for doc in results:
        context = context + context_template.format(question=doc["question"], text=doc["text"]) + "\n\n"

    prompt = prompt_template.format(question=query, context=context).strip()

    return prompt
# %%
prompt = build_prompt(docs)
# %%
print(prompt)

# %%
import tiktoken
encoding = tiktoken.encoding_for_model("gpt-4o")

# %%
len(encoding.encode(prompt))
# %%
