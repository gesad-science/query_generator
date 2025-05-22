from langchain_community.document_loaders import CSVLoader
from langchain_text_splitters import TokenTextSplitter
import numpy as np
import random
import requests
import json
import csv
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

#Evolution prompt templpates
multi_context_template = """
I want you to rewrite the given `input` so that it requires readers to use information from al elements in `Context`.

1. `Input` should require information from all `Context` elements.
2. `Rewritten Input` must be concise and fully answerable from `Context`.
3. Do not use phrases like 'based on the provided context.'
4. `Rewritten Input` should not exceed 15 words.

Context: {context}
Input: {original_input}
Rewritten Input:
"""
reasoning_template = """
I want you to rewrite the given `input` so that it explicity requests multi-step reasoning.

1. `Rewritten Input` should require multiple logical connections or inferences.
2. `Rewritten Input` should be concise and understandable.
3. Do not use phrases like 'based on the provided context.'
4. `Rewritten Input` must be fully answerable from `Context`.
5. `Rewritten Input` should not exceed 15 words.

Context: {context}
Input: {original_input}
Rewritten Input:
"""
hypothetical_scenario_template = """
I want you to rewrite the given `input` to incorporate a hypothetical or speculative scenario.

1. `Rewritten Input` should encourage applying knowledge from `Context` to deduce outcomes.
2. `Rewritten Input` should be concise and understandable.
3. Do not use phrases like 'based on the provided context.'
4. `Rewritten Input` must be fully answerable from `Context`.
5. `Rewritten Input`should not exceed 15 words.

Context: {context}
Input: {original_input}
Rewritten Input:
"""
evolution_templates = [multi_context_template, reasoning_template, hypothetical_scenario_template]

#Function to acess models thorugh chat
def ollama_chat(prompt, model="mistral"):
    url = "http://localhost:11434/api/generate"
    response = requests.post(url, json={
        "model":model,
        "prompt":prompt,
        "stream": False
    })
    response.raise_for_status()
    return response.json()["response"]

#Function to convert to Embeddings
def ollama_embeddings(texts, model="nomic-embed-text"):
    url = "http://localhost:11434/api/embeddings"
    embeddings = []
    for text in texts:
        response = requests.post(url, json={"model":model, "prompt":text})
        response.raise_for_status()
        embedding = response.json()["embedding"]
        embeddings.append(embedding)
    return embeddings

#Function to peform random evolution steps
def evolve_query(original_input, context, steps):
    current_input = original_input
    for _ in range(steps):
        chosen_template = random.choice(evolution_templates)
        evolved_prompt = chosen_template.replace("{context}", str(context)).replace("{original_input}", current_input)
        current_input = ollama_chat(evolved_prompt)
    return current_input

#Chunk Document
input_path = "Input/dataset_tratado.csv"
text_splitter = TokenTextSplitter(chunk_size=1024, chunk_overlap=0)
loader = CSVLoader(input_path, encoding="utf-8")
raw_chunks = loader.load_and_split(text_splitter)

#Convert chunks into embeddings
content = [rc.page_content for rc in raw_chunks]
embeddings = ollama_embeddings(content)

#Select a random chunk to use as reference
reference_index = random.randint(0, len(embeddings) - 1)
reference_embedding = embeddings[reference_index]
contexts = [content[reference_index]]

#Identify similarity
similarity_threshold = 0.8
similar_indices = []
for i, embedding in enumerate(embeddings):
    product = np.dot(reference_embedding, embedding)
    norm = np.linalg.norm(reference_embedding) * np.linalg.norm(embedding)
    similarity = product / norm
    if similarity>= similarity_threshold and i != reference_index:
        similar_indices.append(i)

for i in similar_indices:
    contexts.append(content[i])

print(contexts)

#Query Generation
prompt = f"""I want you to act as a dataset generator, given the context below, which is a list of strings, generate a list of 20 JSON objects, each object must have only one key: "input".

The value of "input" should be a realistic question or statement that could be answered or addressed using the context provided.

Important:
- Only return a valid JSON list of 20 objects.
- Each object must look like this: {{ "input": "your question or statement here" }}

Context:
{contexts}"""

query = ollama_chat(prompt)
print(query)

#Evolve Queries
parsed_queries = json.loads(query)
evolved_queries = []
num_evolution_steps = 1 #KEEP IT 1

for item in parsed_queries:
    original_input = item["input"]
    evolved = evolve_query(original_input, contexts, num_evolution_steps)
    print(evolved)
    evolved_queries.append({"original": original_input, "evolved": evolved})

#Export to a CSV file
output_path = "Results/generated_queries.csv"
with open(output_path, mode="w", newline="", encoding="utf-8") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=["original", "evolved"])
    writer.writeheader()
    for row in evolved_queries:
        writer.writerow(row)

print(f"\nAs queries foram salvas em: {output_path}")