import json
import os

import pandas as pd
from datasets import Dataset


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


def prepare_dataset(d):
    print("prepare_dataset")
    return Dataset.from_pandas(pd.DataFrame(d))
