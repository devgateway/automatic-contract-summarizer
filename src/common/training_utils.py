import json
import re
import sys
from collections import defaultdict

from src.common.extraction_utils import html_to_text, pdf_to_text, docx_to_text
from src.common.post_process import string_exists


def predict_from_pdf(pdf_path, prompt, model, tokenizer, max_length_param, device, log_source_file=False,
                     skip_ai=False, chunk_size_characters=1000, normalize_lowercase=False,
                     num_beams=1, do_sample=False, temperature=0.1):
    """
        Using the previously trained model to predict text from a pdf file.
        :param pdf_path: full file path of the pdf file
        :param prompt: same prompt that was used to train the model
        :param model: pretrained model
        :param tokenizer:
        :param max_length_param: Internal parameter for the model, the bigger this number the more content it can process,
        but numbers bigger than 512 will make the inferring process extremely slow.
        :param device: cpu or gpu
        :param log_source_file:
        :param skip_ai: If True, convert the document and skip AI processing.
        :param chunk_size_characters: The number of characters to split the document into chunks
        :param normalize_lowercase: (Experimental) if True, lowercase the input text.
        :param num_beams: A number between 2 and 6 generates good results.
        :param do_sample:
        :param temperature: A number between 0.5 and 1 generates good summaries.
        :return:
        """
    print("###########################################################################################################")
    value = pdf_to_text(pdf_path, log_source_file, normalize_lowercase)

    if not skip_ai:
        return process_text_in_chunks(value, chunk_size_characters, prompt, model, tokenizer, max_length_param, device,
                                      log_source_file, num_beams, do_sample, temperature)


def predict_from_html(html_path, prompt, model, tokenizer, max_length_param, device):
    print("###########################################################################################################")
    print(html_path)
    value = html_to_text(html_path)
    inputs = get_tokenized_input_ids(prompt, value, tokenizer, max_length_param, device)

    # Generate output using the fine-tuned model
    output_text = generate_and_decode(model, inputs, max_length_param, tokenizer)
    print(output_text)
    print("###########################################################################################################")
    return output_text


def predict_from_docx(doc_path, prompt, model, tokenizer, max_length_param, device, log_source_file=False,
                      skip_ai=False, chunk_size_characters=1000, normalize_lowercase=False,
                      num_beams=1, do_sample=False, temperature=0.1):
    """
    Using the previously trained model to predict text from a docx file.
    :param doc_path: full file path of the docx file
    :param prompt: same prompt that was used to train the model
    :param model: pretrained model
    :param tokenizer:
    :param max_length_param: Internal parameter for the model, the bigger this number the more content it can process,
    but numbers bigger than 512 will make the inferring process extremely slow.
    :param device: cpu or gpu
    :param log_source_file:
    :param skip_ai: If True, convert the document and skip AI processing.
    :param chunk_size_characters: The number of characters to split the document into chunks
    :param normalize_lowercase: (Experimental) if True, lowercase the input text.
    :param num_beams: A number between 2 and 6 generates good results.
    :param do_sample:
    :param temperature: A number between 0.5 and 1 generates good summaries.
    :return:
    """
    print("###########################################################################################################")
    value = docx_to_text(doc_path, log_source_file, normalize_lowercase)

    if not skip_ai:
        return process_text_in_chunks(value, chunk_size_characters, prompt, model, tokenizer, max_length_param, device,
                                      log_source_file, num_beams, do_sample, temperature)


def get_tokenized_input_ids(prompt, value, tokenizer, max_length_param, device):
    inputs = tokenizer(prompt + value, return_tensors="pt", max_length=max_length_param, truncation=True,
                       padding="max_length").input_ids
    inputs = inputs.to(device)  # Move input to the correct device
    return inputs


def generate_and_decode(model, inputs, max_new_tokens, tokenizer, num_beams, do_sample, temperature):
    """
    Using the model and the tokenized input ids summarize the information and decode it into text.
    :param model:
    :param inputs:
    :param max_new_tokens:
    :param tokenizer:
    :param num_beams:
    :param do_sample:
    :param temperature:
    :return:
    """
    output_ids = model.generate(inputs, max_new_tokens=max_new_tokens, num_beams=num_beams, do_sample=do_sample,
                                temperature=temperature)
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return output_text


def split_text_into_chunks(text, log_source_file, chunk_size_characters):
    """
    Split long text into smaller chunks. To preserve context from chunk 2 until the end we attach the part of the
    previous chunk.
    :param text:
    :param log_source_file:
    :param chunk_size_characters:
    :return:
    """
    chunks = []
    for i in range(0, len(text), chunk_size_characters):
        chunk = text[i:i + chunk_size_characters]
        if i != 0:
            chunk = text[i - int(
                chunk_size_characters * 0.2):i] + chunk  # Add last chunk_size characters of the previous chunk
        chunks.append(chunk)
        if log_source_file:
            print("--------------Start Chunk--------------")
            print(chunk)
            print("--------------End Chunk--------------")
    return chunks


def detect_data_for_training(chunk, text_output, log_source_file, train_tender_data, train_goods_data):
    """
    Since not every chunk has all the summary data from text_output then we analyze the chunk and look for the values in
    the map, text_output has a structure [key: value]
    :param train_goods_data: If True, then use tender.goods.xxx entries.
    :param train_tender_data: If True, then use any entry that doesn't match tender.goods.xxx
    :param chunk:
    :param text_output:
    :param log_source_file:
    :return:
    """
    chunk_results = {}
    for key, value in parse_data_flat(text_output).items():
        if train_goods_data is False and 'tender.goods' in key:
            continue

        if train_tender_data is False and 'tender.goods' not in key:
            continue

        # We need to ignore numbers that casually match some tender.goods.N.quantity but are not part of the goods list.
        # Also, we need to ignore "units" if there is no description.
        if 'tender.goods.' in key and ('unit' in key or 'quantity' in key):
            goods_index = re.findall(r'.\d+.', key)
            if not goods_index:
                continue
            description_index = 'tender.goods' + goods_index[0] + 'description'
            # I can make this check because every json_ouput defines the description before unit and quantity.
            if chunk_results.get(description_index) is not None:
                if isinstance(value, str) and value in chunk:
                    if value != "":
                        chunk_results[key] = value
                elif isinstance(value, (int, float)) and str(value) in chunk:
                    chunk_results[key] = value
        else:
            if isinstance(value, str) and value in chunk:
                if value != "":
                    chunk_results[key] = value
            elif isinstance(value, (int, float)) and str(value) in chunk:
                chunk_results[key] = value

    # convert to json string
    chunk_results = json.dumps(chunk_results)
    if log_source_file:
        print("--------------Start Chunk--------------")
        print(chunk)
        print(chunk_results)
        print("--------------End Chunk--------------")
    return chunk_results


def integrate_results(results_list):
    integrated_results = {}
    for result in results_list:
        integrated_results.update(result)
    return integrated_results


def parse_data_flat(data):
    """
    Convert the summary text into a list of key-value pairs.
    :param data:
    :return:
    """
    lines = data.strip().split('\n')
    result = {}
    for line in lines:
        if not line.strip():
            continue
        key_value = line.split(':', 1)
        if len(key_value) != 2:
            continue
        key, value = key_value
        key = key.strip()
        value = value.strip().strip('"')
        # Try to convert numeric values to int
        try:
            value = int(value)
        except ValueError:
            pass
        result[key] = value
    return result


def process_text_in_chunks(value, chunk_size_characters, prompt, model, tokenizer, max_length_param, device,
                           log_source_file, num_beams, do_sample, temperature):
    chunks = split_text_into_chunks(value, log_source_file, chunk_size_characters=chunk_size_characters)
    unique_set = set()
    unique_values = set()
    unique_set_with_chunks = set()
    i = 0
    for chunk in chunks:
        inputs = get_tokenized_input_ids(prompt, chunk, tokenizer, max_length_param, device)

        # Generate output using the fine-tuned model
        # Make the max_new_tokens = max_length_param * 2 to be on the safe side.
        max_new_tokens = max_length_param * 2
        output_text = generate_and_decode(model, inputs, max_new_tokens, tokenizer, num_beams, do_sample, temperature)
        matches = re.findall(r'"([^"]+)":\s*(?:"([^"]+)"|([^,\}]+))', output_text)
        for e in matches:
            # To prevent duplicated entries in the set, we need to evaluate the value, not only the key because
            # the list of goods can produce different keys when the index is calculated,
            # ie: tender.goods.1.description: 'Something' and tender.goods.3.description: 'Something'.
            # This is a side effect of splitting in chunks and preserving the previous chunk for context.
            if 'tender.goods.' in e[0] and '.description' in e[0]:
                if e[1] not in unique_values:
                    unique_values.add(e[1])
                    if e[0] not in unique_set:
                        unique_set.add(e[0] + ': ' + e[1])
                        unique_set_with_chunks.add(e[0] + ' (' + str(i) + '): ' + e[1])
            else:
                # I had to combine this 2 values because the new regex sometimes put an extra " in e[1] and the real value in e[2].
                if 'tender.goods' not in e[0]:
                    unique_set.add(e[0] + ': ' + e[1] + e[2])
                    unique_set_with_chunks.add(e[0] + ': ' + e[1] + e[2])
                else:
                    if e[0] not in unique_set:
                        unique_set.add(e[0] + ': ' + e[1] + e[2])
                        unique_set_with_chunks.add(e[0] + ' (' + str(i) + '): ' + e[1] + e[2])

        i += 1
        sys.stdout.write(f"\rChunk {i} out of {len(chunks)}")
        sys.stdout.flush()
    sys.stdout.write(f"\r")
    sys.stdout.flush()
    unique_set_with_chunks = remove_hallucinated_data(unique_set_with_chunks, value, log_source_file)
    unique_set_with_chunks = remove_incomplete_goods(unique_set_with_chunks, log_source_file)
    unique_set_with_chunks = remove_substring_values_case_insensitive(unique_set_with_chunks)
    unique_set_with_chunks = sorted(unique_set_with_chunks)
    print(unique_set_with_chunks.__str__().replace("',", '\n'))
    return unique_set_with_chunks


def remove_substring_values_case_insensitive(str_list):
    # Step 1 & 2: Parse and group by key
    groups = defaultdict(list)
    for s in str_list:
        key, value = s.split(":", 1)
        key = key.strip()
        value = value.strip()
        groups[key].append(value)

    # Step 3: Filter out values that are substrings of others, ignoring case
    filtered_strings = []
    for key, values in groups.items():
        keepers = []
        for v in values:
            v_lower = v.lower()
            # If `v` is a substring of any other `v2`, ignoring case, skip it
            # Check both substring (v_lower in v2_lower) AND ensure they're not the same ignoring case
            if any(v_lower in other_val.lower() and v_lower != other_val.lower()
                   for other_val in values if other_val != v):
                continue
            keepers.append(v)

        # Step 4: Rebuild the "key: value" strings
        for val in keepers:
            filtered_strings.append(f"{key}: {val}")

    return filtered_strings


# Remove values that are not present in the original document.
def remove_hallucinated_data(data_set, original_data, log_source_file):
    data_set_copy = data_set.copy()
    for data in data_set:
        if ":" in data:
            val = data.split(":")[1].strip()
            if not string_exists(original_data, val, fuzzy_threshold=85, soundex_distance_threshold=1):
                data_set_copy.remove(data)
                if log_source_file:
                    print(f"Not Found: {val}")

            # val = data.split(":")[1].strip().lower().replace('"', '').replace("'", '')
            # if val not in original_data.lower().replace('"', '').replace("'", ''):
            #    if log_source_file:
            #        print(f"Not Found: {val}")
            #     data_set_copy.remove(data)

    return data_set_copy


# This function was added as a last step to try and make the list of goods more "consistent"; one problem we found was
# that processing documents in chunks and reusing the previous page for context created 2 problems:
# 1) duplicated goods with a different key but same value and 2) missing good names and orphan units and quantities.
# Also, many documents had several lists in it (in the same chunk or in different chunks); sometimes the correct list
# of goods was duplicated in the document, in other cases there were lists with good sub-details that polluted the
# extracted data.
def remove_incomplete_goods(data_set, log_source_file):
    new_dataset = set()
    for data in data_set:
        if 'tender.goods.' in data:
            if '.unit' in data or '.quantity' in data:
                chunk = re.findall(r'\(\d+\)', data)
                index = re.findall(r'\.\d+\.', data)
                if index.__len__() > 0:
                    if 'tender.goods' + index[0] + 'description ' + str(chunk[0]) in "".join(data_set):
                        new_dataset.add(data)
                    else:
                        if log_source_file:
                            print(f"Removed: {data}")
                else:
                    if log_source_file:
                        print(f"Removed: {data}")
            else:
                new_dataset.add(data)
        else:
            new_dataset.add(data)

    return new_dataset
