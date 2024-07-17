## 3.2 Semantic Search with Elasticseachr

Use the same docker container and command to run it as shown in module 1:

```
docker run -it \
    --rm \
    --name elasticsearch \
    -p 9200:9200 \
    -p 9300:9300 \
    -e "discovery.type=single-node" \
    -e "xpack.security.enabled=false" \
    docker.elastic.co/elasticsearch/elasticsearch:8.4.3
```

1. Create documents (from documents.json)
2. Use a pretrained model (from Hugging Face) to get the embeddings of the documents
3. Index embeddings into elastic search database

- A document is a collection of fields with their associated values.
- To work with ElasticSearch you have to organize your data into documents, and then add all your documents to an index.

* index is a collection of documents that is stored in a highliy optimized format designed to perform efficient searches.

## 3.3 Evaluation Retrieval

- There are differentways of retrieving the data for a RAG system (e.g. minsearch, elasticsearch, etc.)
- The best way of doing this depends on the data and environments
- For that we use evaluation metrics

Plan for this section:

- Why do we need evaluation
- Evaluation metrics
- Ground truth / gold standard data
- Generating ground truth with LLM
- Evaluating the search results

Query: I just discovered the course, Can I still join?
Relevant documents: doc1, doc10, doc11

For each query we know the releaant items (documents). We can use different evaluation metrics to understand, if the system is performing well.

In general, we have several relevant documents for a single query, for this video, we assume, that we have only one relevant document for each query, which is a simplification of the real problem.

Query: I just discovered the course, Can I still join?
Relevant documents: doc1

ToDo:

for each question in FAQ:
generate 5 questions

For that we will use an LLM (there are other methods to do this, e.e. using humans, domain experts, etc). That is if we have a dataset of 1000 documents, we create 5000 queries.

**Evaluation metrics used here**

- Hit Rate (HR): measures the proportion of quieries for which at least one relevant document is retrieved in the top k(=5) results
- Mean Reciprocal Rank (MRR): Evaluates the rank position of the first relevant document

1. Evaluate text search
2. evaluate vector search
   - Evaluate 3 different methods: Ranking with question, answer, question+answer embeddings
