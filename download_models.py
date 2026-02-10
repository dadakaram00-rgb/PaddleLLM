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

        tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        tokenizer.save_pretrained(llm_path)

        # Load model with float32 for broad CPU compatibility
        model = AutoModelForCausalLM.from_pretrained(llm_model_name, dtype="float32")
        model.save_pretrained(llm_path)
        print(f"LLM saved to {llm_path}")
    except Exception as e:
        print(f"Failed to download LLM: {e}")
        # Fallback to a different model if needed, or just exit
        return

    # 2. Download and Save OCR Models
    print("Downloading PaddleOCR models (German supported)...")
    try:
        # Initialize PaddleOCR to trigger the download of default models.
        # We specify 'de' for German language support.
        # use_gpu=False ensures we get models that don't strictly require GPU (though models are usually same)
        ocr = PaddleOCR(use_angle_cls=True, lang='de', use_gpu=False, show_log=False)

        # The models are downloaded to ~/.paddleocr by default.
        # We need to copy them to our local ./models/ocr directory for offline portability.
        home = os.path.expanduser("~")
        src_ocr = os.path.join(home, ".paddleocr")
        dst_ocr = "./models/ocr"

        if os.path.exists(dst_ocr):
            print(f"Removing existing {dst_ocr}...")
            shutil.rmtree(dst_ocr)

        if os.path.exists(src_ocr):
            print(f"Copying OCR models from {src_ocr} to {dst_ocr}...")
            shutil.copytree(src_ocr, dst_ocr)
            print(f"OCR models copied successfully.")
        else:
            print("Error: ~/.paddleocr directory not found. Models might not have downloaded.")

    except Exception as e:
        print(f"Failed to download/copy OCR models: {e}")

if __name__ == "__main__":
    download_models()
