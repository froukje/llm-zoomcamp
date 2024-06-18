Original Repo: https://github.com/alexeygrigorev/build-your-own-search-engine

# RAG

- R = Search (Retrieval)
- RAG Workflow:
  - User Question send to database
  - Get n most relevant documents
  - Build a prompt based on these documents
    - Question: {Q1}
      Context:
      Answer:
  - Send this prompt to LLM
  - The LLM replies with the answer

* Important part of this is the search of the most relevant documents
* Search
  - Text search
    - find exact matches of words in the query
    - synonyms are not found
  - Semantic / Vector search
    - the query is encoded as vectors

## Practical Implementation Aspects and Tools

- Real-world implementation tools:
  - inverted indexes for text search
  - LSH (locality-sensitive hashing) for vector search (using random projections)
    - fast implementation: random projection (sklearn)
    - book: mining massive datasets (old, but good)
- Technologies:
  - Lucene/Elasticsearch for text search
  - FAISS and and other vector databases
