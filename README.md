# AI Health Insurance Claims Pipeline

An end-to-end insurance claims workflow that turns unstructured claim text into structured fields, validates the result against deterministic policy rules, and fine-tunes a local LoRA adapter for extraction.

The repository combines three pieces:

- Document ingestion and rule extraction from insurance PDFs
- Training-data preparation and LoRA fine-tuning on claim examples
- Claim extraction plus policy validation at inference time

## What the Project Does

The demo pipeline reads a free-form claim description such as a hospital note or email, extracts structured fields like `plan_id`, `age`, `claim_amount`, and `condition_keywords`, then checks those fields against the structured insurance rules stored in the SSOT JSON file.

The overall flow is:

1. Extract policy details from raw PDFs into a detailed CSV.
2. Convert the policy CSV into a structured rules JSON file.
3. Convert raw claim data into instruction-style training examples.
4. Fine-tune a local `microsoft/Phi-3-mini-4k-instruct` adapter with LoRA.
5. Run the demo in `main.py` to extract and validate a claim.

## Repository Layout

- [main.py](main.py): Demo entrypoint that loads the LoRA adapter and validation engine.
- [src/inference.py](src/inference.py): Local LLM wrapper for structured claim extraction.
- [src/engine.py](src/engine.py): Deterministic policy validation engine.
- [src/train.py](src/train.py): LoRA fine-tuning script.
- [src/prepare_data.py](src/prepare_data.py): Builds instruction-style claim training data.
- [src/ingest_pdf.py](src/ingest_pdf.py): Uses Gemini and PDF parsing to build detailed insurance plan data.
- [src/rulesGenerator.py](src/rulesGenerator.py): Converts detailed plan data into the SSOT rules JSON.
- [data/raw/](data/raw): Raw datasets and source documents.
- [data/processed/](data/processed): Generated training data.
- [data/rules/](data/rules): Structured policy rules used by the validation engine.
- [models/lora_adapter/](models/lora_adapter): Saved adapter and tokenizer artifacts.

## Requirements

- Python 3.11 or newer is recommended.
- A CUDA-capable GPU is strongly recommended for inference and training.
- Hugging Face model access for `microsoft/Phi-3-mini-4k-instruct`.
- A Gemini API key if you want to run the PDF policy extraction script.

The core Python dependencies are listed in [requirements.txt](requirements.txt).

## Setup

From the repository root on Windows PowerShell:

```powershell
.\poc_env\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If you are using a different virtual environment, activate that environment instead of `poc_env`.

## Data Pipeline

### 1. Extract detailed insurance plan data from PDFs

This step scans `data/raw/**/*.pdf`, extracts text with `pdfplumber`, and uses Gemini to produce a detailed CSV.

```powershell
python src\ingest_pdf.py
```

Output:

- `data/raw/Insurance_Plan_Details_Detailed.csv`

Important note:

- `src/ingest_pdf.py` currently contains a hardcoded Gemini API key. Replace that with your own secure configuration before running it in a real environment.

### 2. Build the structured rules SSOT

This step converts the detailed plan CSV into a deterministic JSON rules file consumed by the validation engine.

```powershell
python src\rulesGenerator.py
```

Output:

- `data/rules/Structured_Insurance_Rules.json`

### 3. Prepare claim training data

This step reads the raw claims dataset and the structured rules file, then writes instruction-style JSONL samples for supervised fine-tuning.

```powershell
python src\prepare_data.py
```

Output:

- `data/processed/train_data.jsonl`

### 4. Train the LoRA adapter

This trains the local adapter on the generated instruction dataset.

```powershell
python src\train.py
```

Output:

- `models/lora_adapter/`

## Run the Demo

Once the rules file and adapter exist, run the end-to-end demo:

```powershell
python main.py
```

The demo will:

- Load the LoRA adapter from `models/lora_adapter/`
- Load the SSOT rules from `data/rules/Structured_Insurance_Rules.json`
- Extract structured claim data from a sample claim
- Validate the extracted claim against policy eligibility and exclusions
- Print an approval or rejection decision with a score and reason

## How the Validation Works

The validation engine in [src/engine.py](src/engine.py) applies deterministic checks against the SSOT:

- It verifies the `plan_id` exists in the rules file.
- It checks age against the plan's `min_age` and `max_age` values.
- It rejects claims if extracted condition keywords match policy exclusions.
- If no rule is violated, the claim is approved.

This design keeps the final decision explainable even when the extraction step uses an LLM.

## How the Extraction Model Works

The extraction model in [src/inference.py](src/inference.py) uses:

- `microsoft/Phi-3-mini-4k-instruct`
- 4-bit quantization via BitsAndBytes
- A LoRA adapter loaded from `models/lora_adapter/`
- Greedy decoding to keep the output close to the expected JSON template

The extraction prompt asks the model to return only JSON with this shape:

```json
{
  "plan_id": "",
  "age": 0,
  "claim_amount": 0,
  "condition_keywords": []
}
```

## Expected Inputs and Outputs

### Raw claims dataset

`src/prepare_data.py` expects a CSV named:

- `data/raw/indian_health_insurance_claims_dataset.csv`

### Rules file

`src/engine.py`, `main.py`, and `src/prepare_data.py` depend on:

- `data/rules/Structured_Insurance_Rules.json`

### Adapter artifacts

`main.py` and `src/inference.py` expect:

- `models/lora_adapter/`

## Troubleshooting

- If `main.py` says the rules file is missing, run `python src\rulesGenerator.py` first.
- If `main.py` says the adapter is missing, run `python src\train.py` first.
- If the extraction script fails on GPU setup, check CUDA, PyTorch, and BitsAndBytes compatibility.
- If the JSON output from the model is malformed, inspect the prompt and the adapter artifacts in `models/lora_adapter/`.
- If the PDF ingestion script fails, confirm the Gemini API key is configured and the input PDFs exist under `data/raw/`.

## Security Notes

- Do not commit real API keys or tokens into source files.
- Move secrets into environment variables or a local `.env` file if you extend the project.
- Treat generated policy data as derived content and validate it before using it for production decisions.

## Suggested Next Steps

- Replace the hardcoded Gemini key with environment-based configuration.
- Add a small automated test suite for the validation engine.
- Add a CLI wrapper or task runner for the full pipeline.

## License

No license file is present in the repository yet. Add one before distributing the project publicly.