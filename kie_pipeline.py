import os
import argparse
import paddle
from paddleocr import PaddleOCR
from paddlenlp.transformers import AutoModelForCausalLM, AutoTokenizer
import json

def find_model_dir(base_path, model_type, preferred_lang=None):
    """
    Recursively search for a directory containing specific model files
    (inference.pdmodel) within the base path, filtering by model type keywords.
    """
    candidates = []
    for root, dirs, files in os.walk(base_path):
        if "inference.pdmodel" in files:
            # Check if path looks like the right model type
            if model_type in root:
                candidates.append(root)

    if not candidates:
        return None

    # Filter by preferred language if specified
    if preferred_lang:
        for c in candidates:
            if preferred_lang in c:
                return c

    # Fallback to the first candidate found if preference not met
    return candidates[0]

def load_ocr_model(models_dir="./models/ocr"):
    """
    Load PaddleOCR with models from local directory.
    """
    print("Loading OCR models from local storage...")

    # We expect the models to be in ./models/ocr (which mimics ~/.paddleocr structure)
    # Typically: ./models/ocr/whl/det/en/en_PP-OCRv3_det_infer

    det_dir = find_model_dir(models_dir, "det")
    # For recognition, prefer German ("de") models
    rec_dir = find_model_dir(models_dir, "rec", preferred_lang="de")
    cls_dir = find_model_dir(models_dir, "cls")

    if not det_dir or not rec_dir or not cls_dir:
        print(f"Warning: Could not find all OCR model directories in {models_dir}.")
        print(f"Det: {det_dir}, Rec: {rec_dir}, Cls: {cls_dir}")
        print("Attempting to use default PaddleOCR (might require internet if not found)...")
        # Fallback to default, hoping it finds them in ~/.paddleocr if online or cached
        return PaddleOCR(use_angle_cls=True, lang='de', use_gpu=False, show_log=False)

    print(f"Using Det Model: {det_dir}")
    print(f"Using Rec Model: {rec_dir}")
    print(f"Using Cls Model: {cls_dir}")

    ocr = PaddleOCR(
        det_model_dir=det_dir,
        rec_model_dir=rec_dir,
        cls_model_dir=cls_dir,
        use_angle_cls=True,
        lang='de',
        use_gpu=False,
        show_log=False
    )
    return ocr

def load_llm_model(model_path="./models/llm"):
    """
    Load Qwen or similar LLM from local directory.
    """
    print(f"Loading LLM from {model_path}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path, dtype="float32")
        model.eval()
        return tokenizer, model
    except Exception as e:
        print(f"Error loading LLM: {e}")
        return None, None

def extract_text_from_image(ocr, image_path):
    print(f"Processing image: {image_path}")
    result = ocr.ocr(image_path, cls=True)

    extracted_text = []
    if result and result[0]:
        for line in result[0]:
            # line structure: [[box], [text, confidence]]
            text = line[1][0]
            extracted_text.append(text)

    full_text = "\n".join(extracted_text)
    return full_text

def perform_kie(tokenizer, model, text):
    print("Performing Key Information Extraction (KIE)...")

    # Prompt Engineering for KIE
    # We ask the model to extract specific fields.
    # Adjust fields based on "German Contract" context.
    prompt = (
        "You are an intelligent assistant. Extract key information from the following German contract text.\n"
        "Extract the following fields if present:\n"
        "- Contract Title (Vertragstitel)\n"
        "- Date (Datum)\n"
        "- Parties Involved (Vertragsparteien)\n"
        "- Total Amount (Gesamtbetrag)\n\n"
        "Input Text:\n"
        f"{text}\n\n"
        "Provide the output in JSON format."
    )

    # Format input for Qwen/Chat models if needed, but raw prompt often works for base models.
    # For Instruct models, we should use the chat template if possible.
    # Here we use a simple formatted string.

    inputs = tokenizer(prompt, return_tensors="pd")
    input_ids = inputs['input_ids']

    # Generate
    outputs = model.generate(
        input_ids,
        max_length=2048,
        decode_strategy="greedy_search", # Deterministic
        repetition_penalty=1.1
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Post-process to try to extract just the JSON part if the model chats
    # (Simple heuristic)
    return response

def main():
    parser = argparse.ArgumentParser(description="Offline KIE Pipeline using PaddleOCR and LLM")
    parser.add_argument("image_path", help="Path to the scanned contract image")
    args = parser.parse_args()

    # Ensure we are using CPU
    paddle.set_device("cpu")

    # Load Models
    ocr = load_ocr_model()
    tokenizer, model = load_llm_model()

    if not ocr or not tokenizer or not model:
        print("Failed to load models. Exiting.")
        return

    # 1. OCR
    text = extract_text_from_image(ocr, args.image_path)
    print("-" * 40)
    print("Extracted Text Preview:")
    print(text[:500] + "..." if len(text) > 500 else text)
    print("-" * 40)

    # 2. KIE with LLM
    kie_result = perform_kie(tokenizer, model, text)

    print("KIE Result:")
    print(kie_result)

if __name__ == "__main__":
    main()
