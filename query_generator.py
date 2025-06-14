from langchain_community.document_loaders import CSVLoader
from langchain_text_splitters import TokenTextSplitter
from halo import Halo
from dotenv import load_dotenv
# import openai
import numpy as np
#import pandas as pd
import random
import requests
import json
import csv
import sys
import io
import re

# OpenAI API Key confguration
# load_dotenv()
# openai.api_key = os.getenv("OPEN_AI_KEY")

# Correction of caracther problems
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
spinner_generation = Halo(text='Generating Queries\n', spinner='dots')
spinner_contexts = Halo(text='Generating Contexts\n', spinner='dots')
spinner_evolution = Halo(text='Evolving Queries\n', spinner='dots')

# Evolution prompt templpates
multi_context_template = """
I want you to rewrite the given `input` so that it requires readers to use 
information from al elements in `Context`.

1. `Input` should require information from all `Context` elements.
2. `Rewritten Input` must be concise and fully answerable from `Context`.
3. Do not use phrases like 'based on the provided context.'
4. `Rewritten Input` should not exceed 15 words.

Context: {context}
Input: {original_input}
Rewritten Input:
"""
reasoning_template = """
I want you to rewrite the given `input` so that it explicity requests multi-step
reasoning.

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
I want you to rewrite the given `input` to incorporate a hypothetical or 
speculative scenario.

1. `Rewritten Input` should encourage applying knowledge from `Context` to 
deduce outcomes.
2. `Rewritten Input` should be concise and understandable.
3. Do not use phrases like 'based on the provided context.'
4. `Rewritten Input` must be fully answerable from `Context`.
5. `Rewritten Input`should not exceed 15 words.

Context: {context}
Input: {original_input}
Rewritten Input:
"""
evolution_templates = [
    multi_context_template, 
    reasoning_template, 
    hypothetical_scenario_template
]

# Convert string to JSON
def context_conversion(text):
    user_msg = re.search(r'user_msg:\s*(.+)', text)
    expected_intent = re.search(r'expected_intent:\s*(.+)', text)
    expected_class = re.search(r'expected_class:\s*(.+)', text)
    expected_attribute = re.search(r'expected_attribute:\s*(.+)', text)
    expected_filtter_attributes = re.search(r'expected_filtter_attributes:\s*(.+)', text)

    result = {
        "user_msg": user_msg.group(1).strip() if user_msg else "",
        "expected_intent": expected_intent.group(1).strip() if expected_intent else "",
        "expected_class": expected_class.group(1).strip() if expected_class else "",
        "expected_attribute": expected_attribute.group(1).strip() if expected_attribute else "",
        "expected_filtter_attributes": expected_filtter_attributes.group(1).strip() if expected_filtter_attributes else ""
    }

    return result
    
# Acess models thorugh chat using ollama
def ollama_chat(prompt, model="gemma3:27b"):
    url = "http://localhost:11434/api/generate"
    response = requests.post(url, json={
        "model":model,
        "prompt":prompt,
        "stream": False
    })
    response.raise_for_status()
    return response.json()["response"]

# Uses OpenAI to acess models and generate text
#def openai_chat(prompt, model="gpt-4o"):
#    response = openai.chat.completions.create(
#        messages=[{"role":"user", "content":prompt}],
#        temperature=0.7
#    )
#    return (response.choices[0].message.content or "").strip()

# Function to convert text to embeddings
def ollama_embeddings(texts, model="nomic-embed-text"):
    url = "http://localhost:11434/api/embeddings"
    embeddings = []
    for text in texts:
        response = requests.post(url, json={"model":model, "prompt":text})
        response.raise_for_status()
        embedding = response.json()["embedding"]
        embeddings.append(embedding)
    return embeddings

# Function to peform random evolution steps
def evolve_query(original_input, context, steps):
    current_input = original_input
    for _ in range(steps):
        chosen_template = random.choice(evolution_templates)
        evolved_prompt = chosen_template.replace("{context}", str(context)).replace("{original_input}", current_input)
        current_input = ollama_chat(evolved_prompt)
    return current_input

# Function to peform context generation
def context_generation(input_path):
    # Chunk Document
    text_splitter = TokenTextSplitter(chunk_size=1024, chunk_overlap=0)
    loader = CSVLoader(input_path, encoding="utf-8")
    raw_chunks = loader.load_and_split(text_splitter)

    # Convert chunks into embeddings
    content = [rc.page_content for rc in raw_chunks]
    spinner_contexts.start()
    embeddings = ollama_embeddings(content)
    spinner_contexts.stop()

    # Select a random chunk to use as reference
    reference_index = random.randint(0, len(embeddings) - 1)
    reference_embedding = embeddings[reference_index]
    contexts = [content[reference_index]]

    # Identify similarity
    similarity_threshold = 0.8
    
    similar_indices = []
    for i, embedding in enumerate(embeddings):
        product = np.dot(reference_embedding, embedding)
        norm = np.linalg.norm(reference_embedding) * np.linalg.norm(embedding)
        similarity = product / norm
        if similarity >= similarity_threshold and i != reference_index:
            similar_indices.append(i)

    for i in similar_indices:
        contexts.append(content[i])
    
    results = []
    for context in contexts:
        result = context_conversion(context)
        results.append(result)

    print(results)
    
    # Save results as JSON file
    output_path = "Results/contexts.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"contexts": results}, f, ensure_ascii=False, indent=2)

    return contexts

# Function to peform query generation using contexts or file examples
def generate_queries(opc, input_path, context=""):
    # Query Generation
    prompt = ""
    i = 42
    if opc == 3 or opc == 5:
        #TODO: Adjust the prompt
        prompt = f"""Bellow there is contexts embeddings, which were generated 
        splitting a dataset of commands sent from user to a chatbot that stores 
        and edits a database, and gruouping the chunks of this dataset throught 
        simillarity on embeddings:

        {context}

        Based on this contexts, generate more 200 simillar commands, whit 
        variations in names, actions, intents and structures, keep in mind the 
        quantity requested.
        """

    if opc == 2:
        cont = 0
        #while cont < 5:
        if cont == 0:
            with open(input_path, 'r', encoding='utf-8') as file:
                dataset = file.read()
        else:
            with open(
                f"Results/generated_queries{i}.csv",
                'r', encoding='utf-8'
                ) as file:
                    dataset = file.read()
        
        cont += 1

        prompt = f"""
        
        """

    spinner_generation.start()
    response = ollama_chat(prompt)
    #response = openai_chat(prompt)
    print(response)
    spinner_generation.stop()

    # Salva a resposta completa em um arquivo CSV
    i += 1
    output_path = f"Results/generated_queries{i}.csv"
    with open(output_path, "w", encoding="utf-8", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["response"])
        writer.writerow([response])

    print(f"\nA resposta foi salva em: {output_path}")
    return output_path

def evolve_queries(contexts, query_output_path):
    with open(query_output_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        generated_queries = [row for row in reader]
    
    evolved_queries = []
    num_evolution_steps = 3

    spinner_evolution.start()
    for item in generated_queries:
        original_input = item.get("response") or item.get("input") or ""
        evolved = evolve_query(original_input, contexts, num_evolution_steps)
        print(evolved)
        evolved_queries.append({"original": original_input, "evolved": evolved})
    spinner_evolution.stop()

    #Export to a CSV file
    output_path = "Results/evolved_queries.csv"
    with open(output_path, mode="w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["original", "evolved"])
        writer.writeheader()
        for row in evolved_queries:
            writer.writerow(row)

    print(f"\nAs queries foram salvas em: {output_path}")


print("============ Query Generator ============")
print("\n1) Generate contexts with embeddings.")
print("2) Generate queries without context (based on CSV).")
print("3) Generate queries based on context.")
print("4) Evolve queries.")
print("5) Complete pipeline.\n")
rsp = int(input("Type the operation: "))

input_path = "Input/dataset_tratado.csv"

if(rsp == 1):
    context_generation(input_path)
if(rsp == 2):
    generate_queries(rsp, input_path)
if(rsp == 3):
    context = context_generation(input_path)
    generate_queries(rsp, input_path, context)
#if(rsp == 4):
    #evolve_queries()
if(rsp == 5):
    context = context_generation(input_path)
    result_path = generate_queries(rsp, input_path, context)
    evolve_queries(context, result_path)
