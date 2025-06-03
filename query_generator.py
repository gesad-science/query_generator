from langchain_community.document_loaders import CSVLoader
from langchain_text_splitters import TokenTextSplitter
from halo import Halo
import numpy as np
import pandas as pd
import random
import requests
import json
import csv
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
spinner_generation = Halo(text='Gerando Queries', spinner='dots')

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
def ollama_chat(prompt, model="llama3:70b"):
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

def context_generation(input_path):
    #Chunk Document
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

    output_path = "Results/contexts.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        for ctx in contexts:
            f.write(ctx.strip())

    return contexts

def generate_queries(input_path, opc, contexts=""):
    # Query Generation
    prompt = ""
    if opc == 3:
        prompt = f"""Bellow there is contexts embeddings, which were generated splitting a dataset of commands sent from user to a chatbot that stores and edits a database, and gruouping the chunks of this dataset throught simillarity on embeddings:

        {contexts}

        Based on this contexts, generate more 200 simillar commands, whit variations in names, actions, intents and structures, keep in mind the quantity requested.
        """

    if opc == 2:
        i = 34
        cont = 0
        #while cont < 5:
        if cont == 0:
            with open(input_path, 'r', encoding='utf-8') as file:
                dataset = file.read()
        else:
            with open(
                f"Results/generated_queries{i}.csv",
                'r', encoding='utf-8') as file:
                dataset = file.read()
        
        cont += 1

        prompt2 = f"""Abaixo segue um dataset de conde a coluna X contem as
        mensagens de um usário para um chatbot e a coluna Y os códigos sql
        gerados que traduzem a mensagem do usuário para uma operação no
        banco do sistema. 

        {dataset}

        Com base nesses dados gere 500 linhas de mensagens e seu respectivo
        código na mesma estrutura dos dados fornecidos.

        """

        prompt = f"""Abaixo estão exemplos em um arquivo csv de comandos
        enviados por usuários a um chatbot de que armazena e edita base de
        dados conforme  o usuário pede através de linguagem natural, sem
        utilizar termos técnicos e em inglês, onde a coluna user_msg
        representa a mensagem de um usuário para um chatbot,
        a expected_intent representa a intenção que o usuário tem
        (ADD, READ, DELETE, UPDATE), a expected_class representa a entidade
        ou tabela que o usuário que realizar uma operação, a
        expected_attributes representa os atributos da mensagem e o
        expected_filter_attributes representa os filtros de operações
        update, read e delete, ambos devem ser uma string de JSON como será
        mostrado no dataset:

        {dataset}

        Com base nos exemplos, gere um arquivo csv com 200 comandos gerados a 
        partir dos exemplos, com variações nos nomes, operações, podendo ser
        diferentes domínios e assuntos, mas nenhum comando pode ser igual ao 
        outro ou aos comandos nos exemplos, você deve entregar todos os dados 
        gerados, sem abreviações por meio de "..." ou "Aqui estão os 200 
        comandos que você pediu: ...", deve conter exatamente a quantidade 
        exigida de 200, os comandos não devem possuir numeração antes dele, 
        você deve entregar apenas os comandos no formato de um cvs, sem nenhum
        texto adicional ou expliocativo, exclusivamente os comados.
        """

        prompt1 = f"""Abaixo estão interações de um usuário com chatbot, onde a 
        coluna user_msg é a mensagem do usuário para o chatbot, expected_intent 
        é a intenção do usuário (ADD, READ, DELETE, UPDATE), expected_class é a 
        entidade que o usuário quer realizar uma operação, expected_attributes é
        os atributos da mensagem e expected_filtter_attributes é os filtros das 
        operações update, read e delete, ambos devem ser uma string de JSON.

        {dataset}

        Com base nos exemplos, gere um arquivo csv com mais 200 comandos, com
        variações nos nomes e operações, podendo ser de diferentes domínios, mas
        nenhum comando deve ser igual ao outro, ou igual aos do exemplo.
        """
        spinner_generation.start()
        response = ollama_chat(prompt)
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

def evolve_queries(query):
    parsed_queries = json.loads(query)
    evolved_queries = []
    um_evolution_steps = 1 #KEEP IT 1

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


print("========= Query Generator 1.0 =========")
print("\n1) Gerar contextos com os embeddings.")
print("2) Gerar queries a partir de CSV (sem contextos).")
print("3) Gerar queries a partir dos contextos.")
print("4) Evoluir queries geradas.")
print("5) Fazer contextos, geração e evolução completas.\n")
rsp = int(input("Digite operação: "))

input_path = "Input/dataset_tratado.csv"

if(rsp == 1):
    context_generation(input_path)
if(rsp == 2):
    generate_queries(input_path, rsp)
if(rsp == 3):
    context = context_generation(input_path)
    generate_queries(input_path, rsp, context)