
# AI Contract Summarizer
### Summary
This project uses open-source LLM libraries to summarize .pdf and .docx documents with information about contracts: tenders data, dates, lists of goods and services, awards data, etc. Its purpose is to help individuals and organizations to process large amounts of contracts in a systemic way, the summary of each document follows the same structure or template that can be easily loaded into a database.

We defined the template as a list of key:value pairs with these options:
```
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
```
Its important to mention that this template can be adapted or modified to fit the necessities of your project.

One of the main goals of the project was to provide code for fine-tuning, testing and processing that can run on consumer grade hardware, like a RTX 2000/3000/4000 series graphics card, this means using as little GPU memory as possible. For that we selected models in different sizes that can fit into 8/12GB of VRAM.
If you have newer, bigger GPU or a cloud server then this code can be scaled up to use larger LLM models with more parameters (Billions instead of Millions).



## How to use the code

### Folder structure
Inside directory /src we have several folders:
- **common:** util classes used to preprocess documents, cleanup outputs, unify results, connect to a database, etc.
- **facebook / google folders:** each one contains the code to fine-tune and test a different model, in our case we found the best relation between **model size & accuracy** in **google_flan_t5_base** and that's the model we will talk about from now on.

### Libraries needed
This is a python 3 project, after creating a new virtual environment install the libraries with ``` pip install -r requirements.txt ```, also is recommended to install the latest CUDA libraries in order to use the GPU instead of the CPU.

### Train, test and process in bulk

#### Train
- Prepare a directory with the training data, each .pdf and .docx needs a .txt file     with the same name and the summarization. Its important that you copy & paste from the original file including special characters.

- Execute ``` 00_google_flan_T5_base_train_dates.py ```, in our case we got good results and low training time using the scripts from **google_flan_t5_base** folder. This step will re-train the original model to improve how it parses dates.

- Execute ``` 01_google_flan_T5_base_train_model.py ``` to re-train the model from the previous step. This process will read the pairs of documents + texts to learn how to summarize using our template.

#### Test
- We can test single documents by executing ``` 01_google_flan_T5_base_test_one.py ``` or ``` 01_google_flan_T5_base_test_model.py ```. We can change parameters like chunk_size, temperature, etc and test which combination produces the best results.

#### Bulk process
- By executing the script ``` 01_google_flan_T5_base_process_directory.py ``` we can summarize all .pdf and .docx documents in a directory.


---
# Using FLAN-T5 Fine-tuning & Inference Toolkit

This repository contains a **two-stage training pipeline** and a set of **helper utilities** for fine-tuning the open-source **[`google/flan-t5-base`](https://huggingface.co/google/flan-t5-base)** model on domain documents (PDF / DOCX) and generating structured results.
Everything can run **locally** or inside a **CUDA-enabled Docker container**.

---

## ✨ Key Features

* **Stage-0 pre-training on dates** – teaches the model to normalize natural-language dates to ISO-8601.
* **Stage-1 domain fine-tuning** – ingests your own corpus of PDFs/DOCXs and expected outputs.
* **Single-file, curated-batch, or full-directory inference** modes.
* **GPU ready** – all scripts auto-detect `torch.cuda.is_available()`.
* Minimal external dependencies; only `transformers`, `datasets`, and common Python data libraries.

---

## 🗂️ Repository Layout

```
.
├── 00_google_flan_T5_base_train_dates.py       # Stage-0 training
├── 01_google_flan_T5_base_train_model.py       # Stage-1 training
├── 01_google_flan_T5_base_test_one.py          # Inference on one file
├── 01_google_flan_T5_base_test_model.py        # Inference on a curated list
├── 01_google_flan_T5_base_process_directory.py # Batch inference over a folder
├── requirements.txt
```

---

## 🔧 Prerequisites

| Purpose                 | Package                                                 |
| ----------------------- | ------------------------------------------------------- |
| Core ML stack           | **PyTorch ≥ 2.0** (with CUDA if you have an NVIDIA GPU) |
| Transformers & datasets | `transformers ≥ 4.40`, `datasets`                       |
| Data helpers            | `pandas`, `python-docx`, `PyPDF2`                       |
| Your project utils      | `src.common.*` must be importable                |

Install locally:

```bash
pip install torch transformers datasets pandas python-docx PyPDF2
```

---

## 🏋️ Training Workflow

### 1. Stage 0 – Date normalization

```bash
python 00_google_flan_T5_base_train_dates.py
```

* Reads `Merged_and_Shuffled_Dates.csv` containing two columns:
  `date_input` → natural date, `expected_output` → ISO date. This is a sample file autogenerated with different date representations and the expected output as DD/MM/YYYY.
* Fine-tunes the base model for 5 epochs and saves to
  `./fine-tuned-model_flan_t5_base_step_0`.

### 2. Stage 1 – Domain fine-tuning

```bash
python 01_google_flan_T5_base_train_model.py
```

* Loads the Stage-0 checkpoint (`fine-tuned-model_flan_t5_base_step_0`).
* Calls `prepare_training_tuples_from_directory()` to build training pairs from
  `DB_DIRECTORY`.
* Trains for 10 epochs (batch = 3) and saves to
  `./fine-tuned-model_flan_t5_base_step_1_<max_len>`.

> **Tip:** Increase `max_length_param` (default 512 tokens) for longer context; if > 1024 the script automatically enables gradient-checkpointing.

---

## 🔍 Inference Modes

| Script                                            | Use case                                         | How to run                                           |
| ------------------------------------------------- | ------------------------------------------------ | ---------------------------------------------------- |
| **`01_google_flan_T5_base_test_one.py`**          | Quick sanity check on one file (path hard-coded) | `python 01_google_flan_T5_base_test_one.py`          |
| **`01_google_flan_T5_base_test_model.py`**        | Evaluate on a curated list of test paths         | `python 01_google_flan_T5_base_test_model.py`        |
| **`01_google_flan_T5_base_process_directory.py`** | Batch-process every PDF/DOCX in a folder         | `python 01_google_flan_T5_base_process_directory.py` |

All three scripts:

* Expect the Stage-1 checkpoint at
  `.../fine-tuned-model_flan_t5_base_step_1_<max_len>`.
  Update the path or pass via environment variables if you reorganise.
* Slice long files into `chunk_size_characters` (\~⅔ of `max_length_param`).
* Use beam-search + temperature sampling (`num_beams = 2`, `temperature = 0.7`).

The directory processor stores results as `<original>_result.txt` and **skips already-processed files**.

---

## 🛠️ Customising & Extending

| What you want                        | Where to change                                               |
| ------------------------------------ | ------------------------------------------------------------- |
| **Different training data location** | `DB_DIRECTORY` (Stage-1)                                      |
| **Different CSV for Stage-0**        | `CSV_FILE` in Stage-0 script                                  |
| **Hyper-parameters**                 | `num_train_epochs`, `learning_rate`, `max_length_param`, etc. |
| **Prompt engineering**               | Update `prompt` variable (all inference scripts)              |
| **Generation strategy**              | `num_beams`, `temperature`, `do_sample` in inference scripts  |

Because each setting is a **Python constant near the top** of the script, you can also refactor to `argparse` if you prefer CLI flags.

---

## 🐛 Troubleshooting

| Symptom                             | Fix                                                                                                |
| ----------------------------------- | -------------------------------------------------------------------------------------------------- |
| `No module named 'src.HC...'`       | Add the project root to `PYTHONPATH` or `pip install -e .` your utilities.                         |
| CUDA present but script runs on CPU | Confirm you invoked Docker with `--gpus all` and that `torch.cuda.is_available()` prints **True**. |
| `FileNotFoundError` on PDFs/DOCX    | Adjust `DB_DIRECTORY`, input paths, or mount the right host folders into Docker.                   |

---

## 🤝 Contributing

Pull requests are welcome! Please open an issue first to discuss significant changes such as:

* Porting scripts to `argparse` or `typer`
* Adding automated evaluation metrics
* Supporting additional document formats

Remember to run `black` and `ruff` before submitting code.

---

## 📄 License

MIT License – see `LICENSE` for details.
