# Offline KIE Pipeline with PaddleOCR and LLM

This project provides a solution for performing Key Information Extraction (KIE) on scanned German contracts using a small LLM (Qwen2-0.5B-Instruct) and PaddleOCR, entirely on CPU and offline.

## Prerequisites

- Python 3.8+
- [PaddlePaddle](https://www.paddlepaddle.org.cn/en) installed.

## Setup Instructions

### Phase 1: Online Setup (On a machine with internet access)

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Download LLM Model:**
    Run the `download_models.py` script. This will download the Qwen2-0.5B LLM.
    ```bash
    python download_models.py
    ```
    This will create a `models/` directory containing:
    - `models/llm/`: The Qwen2-0.5B-Instruct model and tokenizer.

    **Note:** This script does *not* download OCR models. It assumes you already have PaddleOCR models installed in the default location (e.g., `~/.paddleocr` or a custom `.paddelx` folder).

3.  **Prepare for Transfer:**
    Copy the entire project directory (including the `models` folder and `requirements.txt`) to your offline machine.

### Phase 2: Offline Usage (On the target offline machine)

1.  **Install Dependencies:**
    Ensure Python and PaddlePaddle are installed.
    If you haven't installed the python packages yet, you can use the `requirements.txt`. Note that since the machine is offline, you might need to have pre-downloaded the wheel files (`.whl`) for `paddlepaddle`, `paddleocr`, `paddlenlp`, `opencv-python-headless`, and `Pillow`.

2.  **Run the KIE Pipeline:**
    Use the `kie_pipeline.py` script to process an image.
    ```bash
    python kie_pipeline.py path/to/your/contract_image.jpg
    ```

    The script will:
    - Load the LLM model from the local `models/` directory.
    - Load the OCR model from the system default location.
    - Perform OCR on the image to extract text.
    - Use the LLM to extract key fields (Contract Title, Date, Parties, Total Amount).
    - Print the extracted information in JSON format.

## Configuration

- **Models:** The pipeline uses `Qwen/Qwen2-0.5B-Instruct` for KIE and standard `PP-OCRv3` (or v4) models for OCR.
- **Language:** The OCR is configured to prefer German (`lang='de'`) models.
- **Hardware:** The script explicitly sets `paddle.set_device("cpu")` to run efficiently on CPU.

## Troubleshooting

- **Model Not Found:** Ensure the `models` directory contains the `llm` subdirectory. For OCR errors, check that your default PaddleOCR models are correctly installed (e.g. in `~/.paddleocr`).
- **Memory Issues:** The 0.5B model is very small, but if you encounter OOM errors, ensure no other heavy processes are running.
