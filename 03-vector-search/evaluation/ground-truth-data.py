# %%
# We use the same documents as before
# But we need to create ids for each document, so that we can identify them uniquely
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
import hashlib

def generate_document_id(doc):
    # combined = f"{doc['course']}-{doc['question']}"
    combined = f"{doc['course']}-{doc['question']}-{doc['text'][:10]}"
    hash_object = hashlib.md5(combined.encode())
    hash_hex = hash_object.hexdigest()
    document_id = hash_hex[:8]
    return document_id
# %%
for doc in documents:
    doc['id'] = generate_document_id(doc)
# %%
documents[3]
# %%
# check how unique our ids are
from collections import defaultdict
# %%
hashes = defaultdict(list)

for doc in documents:
    doc_id = doc['id']
    hashes[doc_id].append(doc)
# %%
len(hashes), len(documents)
# %%
# which documents don't have unique ids?
for k, values in hashes.items():
    if len(values) > 1:
        print(k, len(values))
# %%
# the documents with the same ids are duplucated questions
hashes['593f7569']
# %%
# save as json
import json
# %%
with open('documents-with-ids.json', 'wt') as f_out:
    json.dump(documents, f_out, indent=2)
# %%
!head documents-with-ids.json
# %%
# use LLM to generate user questions for each document
prompt_template = """
You emulate a student who's taking our course.
Formulate 5 questions this student might ask based on a FAQ record. The record
should contain the answer to the questions, and the questions should be complete and not too short.
If possible, use as fewer words as possible from the record. 

The record:

section: {section}
question: {question}
answer: {text}

Provide the output in parsable JSON without using code blocks:

["question1", "question2", ..., "question5"]
""".strip()
# %%
from openai import OpenAI
#client = OpenAI()
client = OpenAI(
    base_url='http://localhost:11434/v1/',
    api_key='ollama',
)
# %%

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

torch.random.manual_seed(0)
# %%

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-128k-instruct",
    device_map="cpu",
    torch_dtype="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct")
# %%
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)
# %%
generation_args = {
    "max_new_tokens": 500,
    "return_full_text": False,
    "temperature": 0.0,
    "do_sample": False,
}
# %%

def generate_questions(doc):
    prompt = prompt_template.format(**doc)

    messages = [
    {"role": "user", "content": prompt}
    ]

    generation_args = {
        "max_new_tokens": 500,
        "return_full_text": False,
        "temperature": 1.0,
        "do_sample": False,
    }

    output = pipe(messages, **generation_args)
    output = output[0]['generated_text'].strip()
    print(output)

    #json_response = response.choices[0].message.content
    return output #json_response
# %%
generate_questions(doc)
# %%
from tqdm.auto import tqdm
# %%
results = {}
# %%
for doc in tqdm(documents): 
    doc_id = doc['id']
    if doc_id in results:
        continue

    questions = generate_questions(doc)
    results[doc_id] = questions
# %%
# Load already stored (downloaded) results
import pickle
# %%
!ls
# %%
with open('evaluation/results.bin', 'rb') as f_in:
    results = pickle.load(f_in)
# %%
results['1f6520ca']
# %%
parsed_results = {}

for doc_id, json_questions in results.items():
    parsed_results[doc_id] = json.loads(json_questions)
# %%
#print(json_questions)
# %%
#[
#r"How can I resolve the Docker error 'invalid mode: \Program Files\Git\var\lib\postgresql\data'?",
#"What should I do if I encounter an invalid mode error in Docker on Windows?",
#"What is the correct mounting path to use in Docker for PostgreSQL data on Windows?",
#"Can you provide an example of a correct Docker mounting path for PostgreSQL data?",
#r"How do I correct the mounting path error in Docker for \Program Files\Git\var\lib\postgresql\
#    data'?"
#]
# %%
#results[doc_id] = json.dumps(json_questions)
# %%
doc_index = {d['id']: d for d in documents}
# %%
final_results = []

for doc_id, questions in parsed_resulst.items():
    course = doc_index[doc_id]['course']
    for q in questions:
        final_results.append((q, course, doc_id))
# %%
final_results
# %%
import pandas as pd
# %%
df = pd.DataFrame(final_results, columns=['question', 'course', 'document'])

# %%
df.to_csv('ground-truth-data.csv', index=False)
# %%
!head ground-truth-data.csv
# %%
