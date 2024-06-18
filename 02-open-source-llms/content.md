# Open-Source LLMs

## 2.1 Open-Source LLMs - Introduction

- Replacing the LLM box in the RAG flow

## 2.2 Using a GPU in Saturn Cloud

- Registering in Saturn Cloud (we can get a technical demo)
- Configuring secrets and git
- Creating an instance with a GPU

- alternatively google colab can be used

`pip install -U transformers accelerate bitsandbytes`

- links
  - https://saturncloud.io/
  - https://github.com/DataTalksClub/llm-zoomcamp-saturncloud

## 2.3 Flan-T5

- model: `google/flan-t5-xl`
- Notebook: https://github.com/DataTalksClub/llm-zoomcamp/blob/main/02-open-source/huggingface-flan-t5.ipynb

Explanation of Parameters:

- `max_length`: Set this to a higher value if you want longer responses. For example, max_length=300.
- `num_beams`: Increasing this can lead to more thorough exploration of possible sequences. Typical values are between 5 and 10.
- `do_sample`: Set this to True to use sampling methods. This can produce more diverse responses.
- `temperature`: Lowering this value makes the model more confident and deterministic, while higher values increase diversity. Typical values range from 0.7 to 1.5.
- `top_k and top_p`: These parameters control nucleus sampling. top_k limits the sampling pool to the top k tokens, while top_p uses cumulative probability to cut off the sampling pool. Adjust these based on the desired level of randomness.

## 2.4 phi 3

- Model: microsoft/Phi-3-mini-128k-instruct
- Notebook: huggingface-phi3.ipynb
- Links:

  - https://huggingface.co/microsoft/Phi-3-mini-128k-instruct

## 2.5 Mistral-7B

- Model: mistralai/Mistral-7B-v0.1
- Notebook: huggingface-mistral-7b.ipynb
- ChatGPT instructions for serving

- Links:

  - https://huggingface.co/docs/transformers/en/llm_tutorial
  - https://huggingface.co/settings/tokens
  - https://huggingface.co/mistralai/Mistral-7B-v0.1

## 2.7 Ollama - Running LLMs on a CPU

- The easiest way to run an LLM without a GPU is using Ollama
- Notebook ollama.ipynb

For Linux
´´'
curl -fsSL https://ollama.com/install.sh | sh

ollama start
ollama serve phi3
´´´

We can replace OpenAI with Ollama
´´'
from openai import OpenAI

client = OpenAI(
base_url='http://localhost:11434/v1/',
api_key='ollama',
)
´´´

Run the model:

´´'
ollama run phi3
´´´
We can use phi3 now in the terminal:

´´´
I just discovered the course. Can I still join it
´'´

Now we can add a prompt to customize the output.

Prompt example:

´´'
Question: I just discovered the couse. can i still enrol

Context:

Course - Can I still join the course after the start date? Yes, even if you don't register, you're still eligible to submit the homeworks. Be aware, however, that there will be deadlines for turning in the final projects. So don't leave everything for the last minute.

Environment - Is Python 3.9 still the recommended version to use in 2024? Yes, for simplicity (of troubleshooting against the recorded videos) and stability. [source] But Python 3.10 and 3.11 should work fine.

How can we contribute to the course? Star the repo! Share it with friends if you find it useful ❣️ Create a PR if you see you can improve the text or the structure of the repository.

Are we still using the NYC Trip data for January 2021? Or are we using the 2022 data? We will use the same data, as the project will essentially remain the same as last year’s. The data is available here

Docker-Compose - docker-compose still not available after changing .bashrc This is happen to me after following 1.4.1 video where we are installing docker compose in our Google Cloud VM. In my case, the docker-compose file downloaded from github named docker-compose-linux-x86_64 while it is more convenient to use docker-compose command instead. So just change the docker-compose-linux-x86_64 into docker-compose.

Answer:
´´'

(This prompt can be directly put into the terminal)

- The model can be customized: https://github.com/ollama/ollama

* Use Docker

´´'
docker run -it \
 -v ollama:/root/.ollama \
 -p 11434:11434 \
 --name ollama \
 ollama/ollama
´´´

- Pull the model

´´´
docker exec -it bash
ollama pull phi3
´´´
