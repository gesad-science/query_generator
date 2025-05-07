import csv
import ollama
import pandas as pd
from pathlib import Path

#Lê o documento com as queries reais
df = pd.read_csv("dataset_dome.csv")
queries_reais = df["user_msg"].dropna().tolist()

exemplos = "\n".join(f"-{q}" for q in queries_reais[:10])
prompt = f"""
Abaixo estão exemplos de comandos enviados por usuários a um chatbot de que armazena e edita base de dados conforme o usuário pede:

{exemplos}

Com base nos exemplos, gere mais 15 comandos semelhantes, com variações nos nomes, ações e estruturas, podendo ser de diferentes domínios.
"""

#Passa para o modelo
response = ollama.chat(
    model="gemma",
    messages=[{"role": "user", "content": prompt}]
)

#Coloca em um CSV
content = response['message']['content']
queries = [line.strip("- ").strip() for line in content.splitlines() if line.strip()]

with open("queries_sinteticas.csv", "w", newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["user_query"])
    for query in queries:
        writer.writerow([query])