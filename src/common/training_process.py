import random
from functools import partial

from sklearn.model_selection import train_test_split

from src.common.dataset_utils import prepare_dataset
from src.common.extraction_utils import html_to_text, docx_to_text, pdf_to_text
from src.common.training_utils import split_text_into_chunks, detect_data_for_training


def tokenize_training_data(files, training_texts, prompt, chunk_size_characters, tokenizer, max_length_param,
                           normalize_lowercase=False,
                           log_source_file=False,
                           max_percentage_of_empty_pages=0.2,
                           train_tender_data=True, train_goods_data=True):
    """
    Convert each document, prompt and expected text into tokens that can be used to train the model.
    :param train_goods_data: If True, then use tender.goods.xxx entries.
    :param train_tender_data: If True, then use any entry that doesn't match tender.goods.xxx
    :param files: List of files to process, can be .docx or .pdf
    :param training_texts: List of summarization texts done manually for each file in files
        The template we have created follows this structure:
            contractName: ""
            id: ""
            tender.name: ""
            tender.procedure.law: ""
            tender.address: ""
            tender.phone: ""
            tender.endDate: ""
            tender.validity: ""
            tender.fundsSource: ""
            tender.completionPeriod: ""
            tender.email: ""
            tender.goods.description: ""
            tender.goods.quantity: 0
            tender.goods.unit: ""
            award.name: ""
            award.price: 0
            award.currency: ""
    :param prompt: The 'question' we ask the model.
    :param chunk_size_characters: The number of characters to split the document into chunks
    :param tokenizer:
    :param max_length_param: Internal parameter for the model, the bigger this number the more content it can process,
    but numbers bigger than 512 will make the training process extremely slow.
    :param normalize_lowercase: (Experimental) set to True to normalize the text into lowercase.
    :param log_source_file: Set to True to see more output of the process.
    :param max_percentage_of_empty_pages: A side effect of splitting the document into chunks is many chunks will not have relevant information for the summarization,
    with this parameter we can choose how many of these 'empty' pages we want to keep for training.
    :return: the tokenized documents and tokenized texts.
    """

    temp = []
    for file_path, text_output in zip(files, training_texts):
        if file_path.endswith(".html"):
            text = html_to_text(file_path)
        elif file_path.endswith(".docx"):
            text = docx_to_text(file_path, log_source_file=False, normalize_lowercase=normalize_lowercase)
        elif file_path.endswith(".pdf"):
            text = pdf_to_text(file_path, log_source_file=False, normalize_lowercase=normalize_lowercase)

        chunks = split_text_into_chunks(text, False, chunk_size_characters=chunk_size_characters)
        for chunk in chunks:
            chunk_json_output = detect_data_for_training(chunk, text_output, log_source_file,
                                                         train_tender_data, train_goods_data)
            if normalize_lowercase:
                chunk_json_output = chunk_json_output.lower()
            if chunk_json_output == '{}' and random.random() > max_percentage_of_empty_pages:
                continue
            # concatenate the input text with the chunk.
            temp.append({"input_text": prompt + '\"' + chunk + '\"', "output_json": chunk_json_output})

    # Create 2 datasets using a percentage of the original for testing the accuracy of the model after each train cycle.
    train_data, eval_data = train_test_split(temp, test_size=0.2, random_state=42)

    # Example dataset preparation
    train_dataset = prepare_dataset(train_data)
    eval_dataset = prepare_dataset(eval_data)

    # Use functools.partial to include tokenizer and max_length_param
    preprocess_function_with_params = partial(preprocess_function, tokenizer=tokenizer,
                                              max_length_param=max_length_param)

    tokenized_dataset = train_dataset.map(preprocess_function_with_params, batched=True)
    tokenized_eval_dataset = eval_dataset.map(preprocess_function_with_params, batched=True)

    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    tokenized_eval_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    return tokenized_dataset, tokenized_eval_dataset


def preprocess_function(examples, tokenizer, max_length_param, device="cuda"):
    print("preprocess_function")
    inputs = [ex for ex in examples["input_text"]]
    outputs = [str(ex) for ex in examples["output_json"]]

    # Tokenize and move to the correct device
    model_inputs = tokenizer(inputs, max_length=max_length_param, truncation=True, padding="max_length",
                             return_tensors="pt")

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(outputs, max_length=max_length_param, truncation=True, padding="max_length",
                           return_tensors="pt").input_ids

    # Move inputs and labels to the correct device
    model_inputs = {key: value.to(device) for key, value in model_inputs.items()}
    labels = labels.to(device)

    model_inputs["labels"] = labels
    return model_inputs
