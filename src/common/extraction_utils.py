import logging
import os

import docx2txt
import html2markdown
import mammoth
from bs4 import BeautifulSoup
import re

from pdf2docx import Converter

from src.common.html_utils import simplify_html


def pdf_to_text_with_intermediate_docx(pdf_path, log_source_file=False):
    print(pdf_path)
    docx_file = 'intermediate.docx'
    # Convert PDF to DOCX
    pdf2docx_logger = logging.getLogger('pdf2docx')
    pdf2docx_logger.setLevel(logging.ERROR)
    pdf2docx_logger.handlers = []
    cv = Converter(pdf_path)
    cv.convert(docx_file)  # All pages by default
    cv.close()
    text = docx_to_text(docx_file)
    os.remove(docx_file)
    if log_source_file:
        print(text)
    return text


def pdf_to_text(pdf_path, log_source_file=False, normalize_lowercase=False):
    """
    Convert .pdf file to plain text, including paragraph and tables.
    :param pdf_path:
    :param log_source_file:
    :param normalize_lowercase:
    :return:
    """
    print(pdf_path)
    import fitz  # PyMuPDF

    def extract_table_text(page):
        tables = page.get_text("dict", flags=11)["blocks"]
        table_texts = [""]
        for table in tables:
            if "lines" in table:
                rows = []
                for i, line in enumerate(table["lines"]):
                    row = []
                    for span in line["spans"]:
                        row.append(span["text"])
                    rows.append(row)
                for row in rows:
                    row_text = ' '.join([f'{value}' for i, value in enumerate(row)])
                    table_texts.append(row_text)
        aux_table = '\n'.join(table_texts)
        aux_table += "\n"
        return aux_table

    doc = fitz.open(pdf_path)
    full_text = []
    table_counter = 0
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text("text")
        table_text = "<table" + str(table_counter) + ">" + extract_table_text(page) + "</table" + str(
            table_counter) + ">"
        table_counter = table_counter + 1
        full_text.append(text)
        if table_text:
            full_text.append(table_text)

    result_text = '\n'.join(full_text)
    result_text = replace_multiple_newlines(result_text)
    result_text = remove_duplicate_paragraphs(result_text)
    if normalize_lowercase:
        result_text = result_text.lower()

    if log_source_file:
        print(result_text)
    return result_text


## Alternative implementation using as it is to extract tables, the result is not great. The only advantage is the tables
# are extracted in the same position in the original text. The tables are not extracted as tables but as text and the "structure" is lost.
def docx_to_text_simple(docx_path, log_source_file=False):
    print(docx_path)
    text = docx2txt.process(docx_path)
    text = replace_multiple_newlines(text)
    print(text)
    return text


def docx_to_text(docx_path, log_source_file=False, normalize_lowercase=False):
    """
    # Convert .docx content into plain text, including paragraph and tables.
    :param docx_path:
    :param log_source_file:
    :param normalize_lowercase:
    :return: plain text
    """
    print(docx_path)
    import docx

    def extract_table_text(table):
        rows = []
        for i, row in enumerate(table.rows):
            row_data = [cell.text.strip() for cell in row.cells]
            rows.append(row_data)
        table_texts = []
        for row in rows:
            row_text = ' '.join([f'{value}' for i, value in enumerate(row)])
            table_texts.append(row_text)
        aux_table = '\n'.join(table_texts)
        aux_table += "\n"
        return aux_table

    doc = docx.Document(docx_path)
    full_text = []
    table_counter = 0
    for element in doc.element.body:
        if element.tag.endswith('p'):
            full_text.append(element.text)
        elif element.tag.endswith('tbl'):
            table = docx.table.Table(element, doc)
            table_text = "<table" + str(table_counter) + ">" + extract_table_text(table) + "</table" + str(
                table_counter) + ">"
            table_counter = table_counter + 1
            if table_text:
                full_text.append(table_text)

    result_text = '\n'.join(full_text)
    result_text = replace_multiple_newlines(result_text)
    result_text = remove_duplicate_paragraphs(result_text)
    if normalize_lowercase:
        result_text = result_text.lower()

    if log_source_file:
        print(result_text)
    return result_text


def pdf_to_html(pdf_path, log_source_file=False):
    print(pdf_path)
    docx_file = 'intermediate.docx'
    # Convert PDF to DOCX
    pdf2docx_logger = logging.getLogger('pdf2docx')
    pdf2docx_logger.setLevel(logging.ERROR)
    pdf2docx_logger.handlers = []
    cv = Converter(pdf_path)
    cv.convert(docx_file)  # All pages by default
    cv.close()
    text = docx_to_html(docx_file)
    os.remove(docx_file)
    if log_source_file:
        print(text)
    return text


def ignore_images(element):
    return ""


options = {
    "convert_image": ignore_images
}


def docx_to_html(file_path, log_source_file=False):
    print(file_path)
    with open(file_path, "rb") as docx_file:
        result = mammoth.convert_to_html(docx_file, **options)
        text = result.value  # The raw text
        text = simplify_html(text)
        if log_source_file:
            print(text)
    return text


def html_to_text(html_path):
    # print("extract_text_from_html")
    with open(html_path, 'rb') as file:
        html_content = file.read()
        # Use BeautifulSoup to parse HTML and extract plain text
        soup = BeautifulSoup(html_content, 'lxml')
        text = soup.get_text()
        # text = text.replace('\n', ' ')
        text = text.replace(",", "")  # To get better results when parsing numbers.
    return text


def extract_fields_from_text(json_data):
    # print("extract_fields_from_text")
    # Extract the contract title, ocid and budget amount from the text
    fields = (json_data.get("releases")[0].get("tender").get("title")
              + '\n' + json_data.get("releases")[0].get("ocid") + '\n'
              + str(json_data.get("releases")[0].get("planning").get("budget").get("amount").get("amount")))
    return fields


def html_to_markdown(html):
    # Convert HTML to Markdown
    markdown = html2markdown.convert(html)
    return markdown


def remove_commas_from_long_dates(text):
    # Define a regular expression pattern to match the date format "Friday, 19th July, 2024"
    pattern = r"\D*\s(\d{1,2})(th|st|nd|rd)\s\D*,?\s(\d{4})\b"

    # Function to remove commas from matched dates
    def remove_commas(match):
        print(match.group(0))
        replaced = match.group(0).replace(",", "")
        print(replaced)
        return replaced

    # Replace commas in all occurrences that match the pattern
    updated_text = re.sub(pattern, remove_commas, text)
    return updated_text


def replace_multiple_newlines(text):
    """
    Replace multiple newlines with a single newline. To use less tokens.
    :param text:
    :return:
    """
    temp_text = re.sub(r' +', ' ', text)
    temp_text = re.sub(r'\n+', '\n', temp_text)
    temp_text = re.sub(r'\n+ +', '', temp_text)
    return temp_text


def remove_duplicate_paragraphs(text):
    """
    A side effect of the pdf/docx conversion to text is sometimes entire paragraphs are duplicated. This function
    will remove duplicate paragraphs.
    :param text:
    :return:
    """
    paragraphs = text.split('\n')
    seen = set()
    unique_paragraphs = []
    for paragraph in paragraphs:
        normalized = paragraph
        if len(normalized) > 15:
            if normalized and normalized not in seen:
                seen.add(normalized)
                unique_paragraphs.append(paragraph)
        else:
            unique_paragraphs.append(normalized)
    return '\n'.join(unique_paragraphs)
