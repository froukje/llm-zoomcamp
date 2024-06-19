# Homework: Introduction

In this homework, we'll learn more about search and use Elastic Search for practice.

## Q1. Running Elastic

Run Elastic Search 8.4.3, and get the cluster information. If you run it on localhost, this is how you do it:

´´´
curl localhost:9200
´´´

**Answer**

Run Elasticsearch

´´'
docker run -it \
 --rm \
 --name elasticsearch \
 -p 9200:9200 \

-p 9300:9300 \
 -e "discovery.type=single-node" \
 -e "xpack.security.enabled=false" \
 docker.elastic.co/elasticsearch/elasticsearch:8.4.
´´'

## Getting the data

Now let's get the FAQ data. You can run this snippet:

´´'
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

´´'

## Q2. Indexing the data

Index the data in the same way as was shown in the course videos. Make the course field a keyword and the rest should be text.

Don't forget to install the ElasticSearch client for Python:

´´'
pip install elasticsearch
´´

Which function do you use for adding your data to elastic?

- insert
- index
- put
- add

**Answer**

index

## Q3. Searching

Now let's search in our index.

For a query "How do I execute a command in a running docker container?", what's the score for the top ranking result?

Use only question and text fields and give question a boost of 4

- 94.05
- 84.05
- 74.05
- 64.05

Look at the ´´'\_score'´´ field.

**Answer**

84.05

## Q4. Filtering

Now let's only limit the questions to machine-learning-zoomcamp.

Return 3 results. What's the 3rd question returned by the search engine?

- How do I debug a docker container?
- How do I copy files from a different folder into docker container’s working directory?
- How do Lambda container images work?
- How can I annotate a graph?

## Q5. Building a prompt

Now we're ready to build a prompt to send to an LLM.

Take the records returned from Elasticsearch in Q4 and use this template to build the context. Separate context entries by two linebreaks `(\n\n)`

```
context_template = """
Q: {question}
A: {text}
""".strip()
```

Now use the context you just created along with the "How do I execute a command in a running docker container?" question to construct a prompt using the template below:

```
prompt_template = """
You're a course teaching assistant. Answer the QUESTION based on the CONTEXT from the FAQ database.
Use only the facts from the CONTEXT when answering the QUESTION.

QUESTION: {question}

CONTEXT:
{context}
""".strip()
```

What's the length of the resulting prompt? (use the len function)

- 962
- 1462
- 1962
- 2462

**Answer**
1462

## Q6. Tokens

When we use the OpenAI Platform, we're charged by the number of tokens we send in our prompt and receive in the response.

The OpenAI python package uses tiktoken for tokenization:

´´'
pip install tiktoken
´´'

Let's calculate the number of tokens in our query:

´´´
encoding = tiktoken.encoding_for_model("gpt-4o")
´´´

Use the encode function. How many tokens does our prompt have?

- 122
- 222
- 322
- 422
