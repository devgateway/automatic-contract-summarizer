import json
import os

import pandas as pd
from datasets import Dataset

from src.common.database_connection import get_contracts_for_training


# Read the directory and return a list of pairs with the original file + the expected output.
def prepare_training_tuples_from_directory(directory):
    texts = []
    files = []
    entries = os.listdir(directory)
    files_list = [file for file in entries if os.path.isfile(os.path.join(directory, file))]
    for file in files_list:
        if file.endswith(".txt"):
            print(f"Processing file: {file}")
            txt_file = os.path.join(directory, file)
            with open(txt_file, 'r', encoding='utf-8') as f:
                text_data = f.read()
                texts.append(text_data)
        elif file.endswith(".docx") or file.endswith(".pdf"):
            print(f"Processing file: {file}")
            file_path = os.path.join(directory, file)
            files.append(file_path)

    return files, texts


# Function to read data from the table and process the files
def prepare_training_tuples_from_db(cnx, directory, country, return_json=True, ignore_files=False):
    rows = get_contracts_for_training(cnx, country, use_text_data=True)
    auxFiles = []
    auxJsons = []
    for row in rows:
        json_data = row[0]  # JSON data in string format
        file_name = row[1]  # Full path to the file
        contract_id = row[2]  # Contract ID
        source = row[3]  # Source name
        text_data = row[4]  # Text data

        file_name = directory + "\\" + source + "\\" + file_name
        # files.append(file_name)

        # Convert the JSON string to a Python dictionary
        try:
            if return_json:
                json_object = json.loads(json_data)
                # print(f"JSON data: {json_object}")
                auxJsons.append(json_object)
            else:
                auxJsons.append(text_data)
            auxFiles.append(file_name)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e} - {contract_id}")
            continue

        # Open and process the file
        if ignore_files is False:
            if os.path.exists(file_name):
                try:
                    with open(file_name, 'r') as file:
                        content = file.read()
                        # print(f"File content from {file_path}:\n{content}\n")
                except OSError as e:
                    print(f"Error opening file {file_name}: {e}")
            else:
                print(f"File does not exist: {file_name}")

    return auxFiles, auxJsons


def prepare_dataset(d):
    print("prepare_dataset")
    return Dataset.from_pandas(pd.DataFrame(d))
