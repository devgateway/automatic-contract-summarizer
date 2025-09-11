from transformers import T5Tokenizer, T5ForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments
import torch

from src.common.dataset_utils import prepare_training_tuples_from_directory
from src.common.training_process import tokenize_training_data

# Replace it with the directory where the documents are stored.
DB_DIRECTORY = r"C:\Git\document-scrapping\data\documents"

# Read from the directory and create 2 lists, for files and texts. The files can be .pdf or .docx, the texts are the
# expected output.
files, training_texts = prepare_training_tuples_from_directory(DB_DIRECTORY)

############################################################################
# Load pre-trained model and tokenizer
model_name = "./fine-tuned-model_flan_t5_base_step_0"  # You can choose a different model as needed
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name, device_map="auto")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Configuration parameters.

# Minimum value is 1, bigger numbers use more VRAM.
per_device_train_batch_size = 2
per_device_eval_batch_size = 2

# Too big epoch doesnt translate in better results.
num_train_epochs = 3  # eval_loss < 0.009.

# Bigger multiplier can process more data at once but it uses more VRAM.
multiplier = 2
max_length_param = 512 * multiplier
if max_length_param > 512 * 3:
    model.gradient_checkpointing_enable()  # Use it with max_length_param > 1024.
prompt = ''
max_percentage_of_empty_pages = 0.1
chunk_size_characters = 512 * multiplier
normalize_lowercase = False
log_source_file = True

# Prepare datasets for training.
tokenized_dataset, tokenized_eval_dataset = tokenize_training_data(files, training_texts, prompt, chunk_size_characters,
                                                                   tokenizer, max_length_param, normalize_lowercase,
                                                                   log_source_file, max_percentage_of_empty_pages)

training_args_seq2 = Seq2SeqTrainingArguments(
    output_dir="./results_google_flan_T5_base",
    eval_strategy="epoch",
    learning_rate=1e-4,
    per_device_train_batch_size=per_device_train_batch_size,
    per_device_eval_batch_size=per_device_eval_batch_size,
    num_train_epochs=num_train_epochs,
    weight_decay=0.01,
    predict_with_generate=True,
    save_strategy="no"
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args_seq2,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    eval_dataset=tokenized_eval_dataset,  # You can split the dataset for validation
)

# Fine-tune the model
print("Start training")
trainer.train()

print("Save the model")
model.save_pretrained("./fine-tuned-model_flan_t5_base_step_1_" + str(max_length_param))
tokenizer.save_pretrained("./fine-tuned-model_flan_t5_base_step_1_" + str(max_length_param))
