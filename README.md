# Query Generator

This programa is used to generate synthetic queries using language models based on user-generated queries documentend on a CSV file.

## Requeriments

1. To run the models locally you need to install and run [Ollama](https://ollama.com/download), and install the desired model through the following command, substituting ``model`` with the desired model.

  ollama pull model

The models used during the tests of this program were Mistral, Llama2, Gemma and Phi.

2. It's necessary a csv file containing user-generated queries on the project folder, if the first cell of the collumn containg the user queries has a different name than ``user_msg``, then you need to change the following name:

  queries_reais = df["user_msg"].dropna().tolist()

substituting 'user_msg' with the name of the column.

3. This program use Ollama and Pandas dependencies, that are proper documentend on the ``requirements.txt`` file.

After running the program, and csv file will be created on the project folder containing the result queries.
