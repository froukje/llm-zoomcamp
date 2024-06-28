# %%
from openai import OpenAI
import openai

client = OpenAI(
    base_url='http://localhost:11434/v1/',
    api_key='ollama',
)
# %%
prompt = "What's the formula for energy?"
# %%
response = client.chat.completions.create(
    model="gemma2b", 
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ],
    temperature=0.0
)
# %%
print(response.choices[0].message.content)
# %%
