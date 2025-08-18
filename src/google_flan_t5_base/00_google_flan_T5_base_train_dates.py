import pandas as pd
from datasets import Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments, Trainer, \
    TrainingArguments
import torch

CSV_FILE = r"..\documents\Merged_and_Shuffled_Dates.csv"

############################################################################
# Load pre-trained model and tokenizer
model_name = "google/flan-t5-base"  # You can choose a different model as needed
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name, device_map="auto")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Load the CSV file into a pandas DataFrame
df = pd.read_csv(CSV_FILE)

# Convert the DataFrame to Hugging Face Dataset
dataset = Dataset.from_pandas(df)

# Display first few rows for verification
print(dataset[0:5])


def preprocess_function(examples):
    print("preprocess_function")
    # Form the input prompt for T5, e.g., "Convert date: 19th July, 2024"
    inputs = ["Convert date: " + ex for ex in examples["date_input"]]
    targets = examples["expected_output"]

    # Tokenize inputs and targets
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")

    # Tokenize the target labels separately as T5 needs target tokenization as well
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=512, truncation=True, padding="max_length")

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


# Apply preprocessing to the dataset
tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    # evaluation_strategy="epoch",
    learning_rate=3e-4,
    per_device_train_batch_size=5,  # the bigger the number, it consumes more VRAM.
    per_device_eval_batch_size=5,  # the bigger the number, it consumes more VRAM.
    num_train_epochs=5,
    weight_decay=0.01,
    logging_dir='./logs'
)

# Initialize the T5 model
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Initialize the Trainer with model, tokenizer, arguments, and datasets
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset,  # Optional: split dataset for separate evaluation
    tokenizer=tokenizer,
)

# Train the model
trainer.train()

# Evaluate the model
evaluation_results = trainer.evaluate()
print("Evaluation Results:", evaluation_results)

print("Save the model")
model.save_pretrained("./fine-tuned-model_flan_t5_base_step_0")
tokenizer.save_pretrained("./fine-tuned-model_flan_t5_base_step_0")
