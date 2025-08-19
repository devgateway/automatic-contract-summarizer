from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

from src.common.training_utils import predict_from_docx, predict_from_pdf

# It's important to match the multiplier used during training.
multiplier = 2
max_length_param = 512 * multiplier
PRE_TRAINED_DIRECTORY = r".\fine-tuned-model_flan_t5_base_step_1_" + str(max_length_param)
chunk_size_characters = int(max_length_param * 0.9)
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
log_source_file = True

# Generation parameters.
num_beams = 2
do_sample = True
temperature = 0.7

############################################################################
# REMEMBER NEVER USE THE SAME FILE FOR TRAINING AND TESTING!!!
new_docx = "new_contract_not_used_for_training.pdf"
predict_from_pdf(new_docx, prompt, model, tokenizer, max_length_param, device, log_source_file=log_source_file,
                 skip_ai=skip_ai, chunk_size_characters=chunk_size_characters, normalize_lowercase=normalize_lowercase,
                 num_beams=num_beams, do_sample=do_sample, temperature=temperature)

new_docx = "new_contract_not_used_for_training.docx"
predict_from_docx(new_docx, prompt, model, tokenizer, max_length_param, device, log_source_file=log_source_file,
                  skip_ai=skip_ai, chunk_size_characters=chunk_size_characters, normalize_lowercase=normalize_lowercase,
                  num_beams=num_beams, do_sample=do_sample, temperature=temperature)
