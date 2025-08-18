import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from pathlib import Path
from src.common.training_utils import predict_from_pdf, predict_from_docx

multiplier = 2
max_length_param = 512 * multiplier
PRE_TRAINED_DIRECTORY = r".\fine-tuned-model_flan_t5_base_step_1_" + str(max_length_param)
chunk_size_characters = int(512 * multiplier * 0.9)
normalize_lowercase = False

# Load pre-trained model and tokenizer
model_name = PRE_TRAINED_DIRECTORY
tokenizer = T5Tokenizer.from_pretrained(model_name, use_fast=True, padding=True, truncation=True)
model = T5ForConditionalGeneration.from_pretrained(model_name, device_map="auto")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
# IMPORTANT: THIS TEXT MUST MATCH THE PROMPT USED FOR TRAINING.
prompt = ''

skip_ai = False
log_source_file = False

# Generation parameters.
num_beams = 2
do_sample = True
temperature = 0.7

# Directory with contracts to be processed (replace it with your own directory)
input_directory = Path(f"C:\Git\document-scrapping\data\documents")
output_directory = Path(f"C:\Git\document-scrapping\data\documents")

# Loop through all .pdf and .docx files, run the model, and save the results
for file_path in input_directory.glob("*.pdf"):
    output_file = output_directory / f"{file_path.stem}_result.txt"
    # Check if .txt file doesnt exists.
    if not output_file.exists():
        result = predict_from_pdf(
            str(file_path),
            prompt,
            model,
            tokenizer,
            max_length_param,
            device,
            log_source_file=log_source_file,
            skip_ai=skip_ai,
            chunk_size_characters=chunk_size_characters,
            normalize_lowercase=normalize_lowercase,
            num_beams=num_beams,
            do_sample=do_sample,
            temperature=temperature
        )
        with output_file.open("w", encoding="utf-8") as f:
            f.write("\n".join(result))
    else:
        print(f"File {output_file} already exists. Skipping.")

for file_path in input_directory.glob("*.docx"):
    output_file = output_directory / f"{file_path.stem}_result.txt"

    # Check if the .txt file doesn't already exist.
    if not output_file.exists():
        try:
            # Attempt to process the .docx file and produce results
            result = predict_from_docx(
                str(file_path),
                prompt,
                model,
                tokenizer,
                max_length_param,
                device,
                log_source_file=log_source_file,
                skip_ai=skip_ai,
                chunk_size_characters=chunk_size_characters,
                normalize_lowercase=normalize_lowercase,
                num_beams=num_beams,
                do_sample=do_sample,
                temperature=temperature
            )

            # Attempt to write the result to the output file
            with output_file.open("w", encoding="utf-8") as f:
                f.write("\n".join(result))

        except Exception as e:
            # Log the error and skip this file; the script continues
            print(f"Error processing file '{file_path}': {e}")

    else:
        print(f"File {output_file} already exists. Skipping.")
