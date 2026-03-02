# Offline KIE Pipeline with PaddleOCR and Transformers

This project provides a solution for performing Key Information Extraction (KIE) on scanned German contracts using ErnieLayout and PaddleOCR, entirely on CPU and offline.

## Project Structure

- `download_models.py`: Run on an **online machine** to download base models.
- `generate_synthetic_data.py`: Creates a synthetic dataset of German contracts for training.
- `train_kie.py`: Fine-tunes ErnieLayout for extraction (Offline & 5GB RAM optimized).
- `infer_pdf.py`: Extracts key-value pairs from a PDF (Offline compatible).
- `data_structure.md`: Documentation on the training data format.

## Offline Workflow

### Phase 1: Online Setup (Internet Required)

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Download Base Models:**
    ```bash
    python download_models.py
    ```
    Creates:
    - `models/llm/`: Qwen2-0.5B-Instruct
    - `models/base_models/ernie-layoutx-base-uncased/`: ErnieLayout Base

3.  **Transfer:** Copy the project to the offline machine.

### Phase 2: Offline Usage

1.  **Generate Data:**
    ```bash
    python generate_synthetic_data.py
    ```

2.  **Fine-tune (CPU):**
    ```bash
    python train_kie.py
    ```
    - Loads: `./models/base_models/ernie-layoutx-base-uncased`
    - Saves: `./models/fine_tuned/ernie_layout_kie`
    - Optimized for **5GB RAM** using SGD optimizer.

3.  **PDF Inference:**
    ```bash
    python infer_pdf.py path/to/document.pdf
    ```
    - Loads: `./models/fine_tuned/ernie_layout_kie`
    - Outputs: `TITLE`, `DATE`, `PARTY`, `AMOUNT`.

## Local Model Paths

- **Base Model:** `./models/base_models/ernie-layoutx-base-uncased`
- **Fine-tuned Model:** `./models/fine_tuned/ernie_layout_kie`
- **OCR Models:** Default system location (e.g., `.paddelx`).

## Hardware Requirements

- **CPU Only.**
- **Memory:** Optimized for **5GB RAM** (Batch Size 1, SGD Optimizer).
