import os
import shutil
import paddle
from paddleocr import PaddleOCR
from paddlenlp.transformers import AutoModelForCausalLM, AutoTokenizer

def download_models():
    print(f"PaddlePaddle version: {paddle.__version__}")

    # 1. Download and Save LLM
    # We use Qwen2-0.5B-Instruct as it is small and supports instructions, good for CPU.
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

        # Load model with float32 for broad CPU compatibility
        model = AutoModelForCausalLM.from_pretrained(llm_model_name, dtype="float32")
        model.save_pretrained(llm_path)
        print(f"LLM saved to {llm_path}")
    except Exception as e:
        print(f"Failed to download LLM: {e}")
        # Fallback to a different model if needed, or just exit
        return

    # 2. OCR Models
    print("Skipping PaddleOCR model download as requested (models assumed present in default location).")

if __name__ == "__main__":
    download_models()
