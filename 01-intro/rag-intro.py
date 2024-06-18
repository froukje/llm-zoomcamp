# %%
# use the search engine implemented in "00-implement-a-sear-engine"
!wget https://raw.githubusercontent.com/alexeygrigorev/minsearch/main/minsearch.py
# %%
import minsearch
import json
# %%
with open("../01-intro/documents.json", "rt") as f_in:
    docs_raw = json.load(f_in)

docs_raw
# %%
documents = []

for course_dict in docs_raw:
    for doc in course_dict['documents']:
        doc['course'] = course_dict['course']
        documents.append(doc)
# %%
documents[0]
# %%
# index the documents
index = minsearch.Index(
    text_fields=["question", "text", "section"],
    keyword_fields=["course"]
)
# %%
# in SQL: SELECT * WHERE course = "data-engineering-zoomcamp"

q = 'the course has already started, can I still enroll?'
# %%
index.fit(documents)
# %%
boost = {"question": 3.0, "section": 0.5}

results = index.search(
    query=q,
    filter_dict = {"course": "data-engineering-zoomcamp"},
    boost_dict=boost,
    num_results=5
)
# %%
results
# %%
# We now can send these most relevant documents to the LLM
from openai import OpenAI
# %%
client = OpenAI()
# %%
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": q}]
)
# %%
# very generic response
response.choices[0].message.content
# %% 
# now use the documents we found
# it is good to give the LLM a role ("teaching assistant")
prompt_template = """
You're a course teaching assistant. 
Answer the QUESTION based on the CONTEXT from the FAQ teamplate. Use only the facts from the CONTEXT, when answering the QUESTION. 
If the CONTEXT doesn't contain the answer, output NONE. 

QUESTION: {question}

CONTEXT: {context}
""".strip()

# %%
context =""

for doc in results:
    context = context + f"section: {doc['section']}\nquestion: {doc['question']}\nanswer: {doc['text']}\n\n"
# %%
print(context)
# %%
prompt = prompt_template.format(question=q, context=context).strip()
# %%
print(prompt)
# %%
# put this prompt into the LLM
# This gives the correct answer
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": prompt}]
)
# %%
# clean the above code
def search(query):
    
    boost = {"question": 3.0, "section": 0.5}

    results = index.search(
    query=query,
    filter_dict = {"course": "data-engineering-zoomcamp"},
    boost_dict=boost,
    num_results=5
    )
    return results

# %%
def build_prompt(query, search_results):
    prompt_template = """
        You're a course teaching assistant. 
        Answer the QUESTION based on the CONTEXT from the FAQ teamplate. Use only the facts from the CONTEXT, when answering the QUESTION. 
        If the CONTEXT doesn't contain the answer, output NONE. 

        QUESTION: {question}

        CONTEXT: {context}
        """.strip()
    
    context =""

    for doc in results:
        context = context + f"section: {doc['section']}\nquestion: {doc['question']}\nanswer: {doc['text']}\n\n"

    prompt = prompt_template.format(question=query, context=context).strip()

    return prompt
# %%
def llm(prompt):
    response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content
# %%
query = "How do I run kafka?"
def rag(query):
    search_results = search(query)
    build_prompt(query, search_results)
    answer = llm(prompt)
    return answer
# %%
# if we now want to use a different search, e.g. elastic search
# we only need to change the "search" function.
# Now replace the "search" function
# run elastic search in docker
# docker run -it \
#    --rm \
#    --name elasticsearch \
#    -p 9200:9200 \
#    -p 9300:9300 \
#    -e "discovery.type=single-node" \
#    -e "xpack.security.enabled=false" \
#    docker.elastic.co/elasticsearch/elasticsearch:8.4.3 
#
# test if elasticsearch runs:
# curl http://localhost:9200
#
# index the documents with elasticsearch
# elasticsearch is persistent, i.e. data is saved on disk
# no new execution is needed (depends how it is run)

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
# query the data
query = "I just discovered the course, can I stin join it?"
def elastic_search(query):
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
                        "course": "data-engineering-zoomcamp"
                    }
                }
            }
        }
    }
    response = es_client.search(index=index_name, body=search_query)

    result_docs = []
    for hit in response["hits"]["hits"]:
        result_docs.append(hit["_source"])

    return result_docs
# %%
elastic_search(query)
# %%
def rag(query):
    search_results = elastic_search(query)
    build_prompt(query, search_results)
    answer = llm(prompt)
    return answer
# %%
rag(query)