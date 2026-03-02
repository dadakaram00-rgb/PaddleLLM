import os
import shutil
import paddle
from paddleocr import PaddleOCR
from paddlenlp.transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    ErnieLayoutForTokenClassification,
    ErnieLayoutTokenizer
)

def download_models():
    print(f"PaddlePaddle version: {paddle.__version__}")

    # 1. Download and Save LLM (Qwen2-0.5B-Instruct)
    llm_model_name = "Qwen/Qwen2-0.5B-Instruct"
    llm_path = "./models/llm"

    print(f"Downloading LLM: {llm_model_name}")
    try:
        if not os.path.exists(llm_path):
            os.makedirs(llm_path)

        try:
            tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        except Exception as e:
            print(f"AutoTokenizer failed ({e}), attempting Qwen2Tokenizer...")
            from paddlenlp.transformers import Qwen2Tokenizer
            tokenizer = Qwen2Tokenizer.from_pretrained(llm_model_name)

        tokenizer.save_pretrained(llm_path)

        model = AutoModelForCausalLM.from_pretrained(llm_model_name, dtype="float32")
        model.save_pretrained(llm_path)
        print(f"LLM saved to {llm_path}")
    except Exception as e:
        print(f"Failed to download LLM: {e}")

    # 2. Download and Save ErnieLayout Base (for offline KIE fine-tuning)
    base_model_name = "ernie-layoutx-base-uncased"
    base_path = "./models/base_models/ernie-layoutx-base-uncased"

    print(f"Downloading ErnieLayout Base: {base_model_name}")
    try:
        if not os.path.exists(base_path):
            os.makedirs(base_path)

        tokenizer = ErnieLayoutTokenizer.from_pretrained(base_model_name)
        tokenizer.save_pretrained(base_path)

        model = ErnieLayoutForTokenClassification.from_pretrained(base_model_name, num_labels=6)
        model.save_pretrained(base_path)
        print(f"ErnieLayout Base saved to {base_path}")
    except Exception as e:
        print(f"Failed to download ErnieLayout Base: {e}")

    # 3. OCR Models
    print("Skipping PaddleOCR model download as requested (models assumed present in default location).")

if __name__ == "__main__":
    download_models()
