from langchain_community.document_loaders import CSVLoader
from langchain_text_splitters import TokenTextSplitter
import numpy as np
import pandas as pd
import random
import requests
import json
import csv
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Função para acessar o modelo generativo por chat
def ollama_chat(prompt, model="mixtral"):
    url = "http://localhost:11434/api/generate"
    response = requests.post(url, json={
        "model":model,
        "prompt":prompt,
        "stream": False
    })

# Abre o arquivo
input_path = "Input/dataset_tratado.csv"
with open(input_path, 'r', encoding='utf-8') as file:
    dataset = file.read()

prompt = """Abaixo estão exemplos em um arquivo csv de comandos enviados por usuários a um chatbot de que armazena e edita base de dados conforme o usuário pede:

{dataset}

Com base nos exemplos, gere um arquivo csv com 500 linhas contendo interações do mesmo modelo que o do exemplo, com variações nos nomes, operações, podendo ser de diferentes domínios e assuntos, mas nenhum comando pode ser igual ao outro, ou iguais aos comandos disponíveis no dataset de exemplo, e não pode conter "..." indicando que existam mais comandos dentro dos três pontos, e entregue exatamente a quantidade que foi pedida.
"""

response = ollama_chat(prompt)
print(response)

output_path = "Results/response.csv"
# Supondo que a resposta seja uma string com comandos separados por nova linha
if response:
    commands = response.strip().split('\n')
    with open(output_path, 'w', encoding='utf-8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['command'])
        for cmd in commands:
            if cmd.strip():
                writer.writerow([cmd.strip()])

