import os
import argparse
import paddle
import pypdfium2 as pdfium
from PIL import Image
from paddleocr import PaddleOCR
from paddlenlp.transformers import ErnieLayoutForTokenClassification, ErnieLayoutTokenizer
import numpy as np

# OFFLINE PATHS: Ensure these exist on your offline PC
FINE_TUNED_MODEL_PATH = "./models/fine_tuned/ernie_layout_kie"
BASE_MODEL_PATH = "./models/base_models/ernie-layoutx-base-uncased"

# PADDLEOCR OFFLINE MODELS (Download and place these in ./models/ocr/)
DET_MODEL_DIR = "./models/ocr/ch_PP-OCRv3_det_infer"
REC_MODEL_DIR = "./models/ocr/ch_PP-OCRv3_rec_infer"
CLS_MODEL_DIR = "./models/ocr/ch_ppocr_mobile_v2.0_cls_infer"

# Label mapping
ID2LABEL = {0: "O", 1: "TITLE", 2: "TITLE", 3: "DATE", 4: "PARTY", 5: "AMOUNT"}

def load_models():
    paddle.set_device("cpu")

    model_to_load = FINE_TUNED_MODEL_PATH
    if not os.path.exists(model_to_load):
        print(f"Warning: Fine-tuned model not found at {model_to_load}. Falling back to base model at {BASE_MODEL_PATH}")
        model_to_load = BASE_MODEL_PATH

    if not os.path.exists(model_to_load):
        raise FileNotFoundError(f"No model found at {model_to_load}. Run download_models.py first.")

    print(f"Loading ErnieLayout model from: {model_to_load}")
    tokenizer = ErnieLayoutTokenizer.from_pretrained(model_to_load)
    model = ErnieLayoutForTokenClassification.from_pretrained(model_to_load)
    model.eval()

    # Initialize PaddleOCR with explicit local model paths for total offline usage
    print("Initializing PaddleOCR with local model paths...")
    # Check if local OCR models exist, otherwise fallback to system default (which assumes pre-installed)
    if os.path.exists(DET_MODEL_DIR):
        ocr = PaddleOCR(
            det_model_dir=DET_MODEL_DIR,
            rec_model_dir=REC_MODEL_DIR,
            cls_model_dir=CLS_MODEL_DIR,
            use_angle_cls=True,
            lang='de',
            use_gpu=False,
            show_log=False
        )
    else:
        print(f"Local OCR models not found at {DET_MODEL_DIR}. Falling back to default system location (e.g., ~/.paddleocr).")
        ocr = PaddleOCR(use_angle_cls=True, lang='de', use_gpu=False, show_log=False)

    return tokenizer, model, ocr

def pdf_to_image(pdf_path, page_index=0):
    pdf = pdfium.PdfDocument(pdf_path)
    page = pdf[page_index]
    bitmap = page.render(scale=2)
    pil_image = bitmap.to_pil()
    return pil_image

def normalize_bbox(bbox, width, height):
    return [
        int(1000 * (bbox[0] / width)),
        int(1000 * (bbox[1] / height)),
        int(1000 * (bbox[2] / width)),
        int(1000 * (bbox[3] / height)),
    ]

def process_document(pdf_path, tokenizer, model, ocr):
    print(f"Processing PDF: {pdf_path}")
    image = pdf_to_image(pdf_path)
    w, h = image.size

    img_array = np.array(image)
    ocr_result = ocr.ocr(img_array, cls=True)

    words = []
    bboxes = []

    if not ocr_result or not ocr_result[0]:
        return {}

    for line in ocr_result[0]:
        text = line[1][0]
        box = line[0]
        words.append(text)
        bboxes.append(normalize_bbox([box[0][0], box[0][1], box[2][0], box[2][1]], w, h))

    vision_image = image.resize((224, 224)).convert("RGB")
    vision_image = np.array(vision_image).transpose(2, 0, 1).astype("float32") / 255.0
    vision_image = paddle.to_tensor([vision_image])

    inputs = tokenizer(
        words,
        bbox=bboxes,
        return_tensors="pd"
    )

    with paddle.no_grad():
        outputs = model(
            input_ids=inputs["input_ids"],
            bbox=inputs["bbox"],
            image=vision_image
        )

    logits = outputs[0] if isinstance(outputs, tuple) else outputs
    predictions = paddle.argmax(logits, axis=-1).numpy()[0]

    # Manual prediction-to-word alignment
    extracted_data = {}
    current_token_idx = 1 # Skip [CLS]

    for i, word in enumerate(words):
        word_tokens = tokenizer.tokenize(word)
        if current_token_idx < 512:
            pred_idx = predictions[current_token_idx]
            label = ID2LABEL.get(pred_idx, "O")

            if label != "O":
                if label not in extracted_data:
                    extracted_data[label] = []
                extracted_data[label].append(word)

            current_token_idx += len(word_tokens)
        else:
            break

    for label in extracted_data:
        extracted_data[label] = " ".join(extracted_data[label])

    return extracted_data

def main():
    parser = argparse.ArgumentParser(description="Offline PDF KIE Inference")
    parser.add_argument("pdf_path", help="Path to the PDF document")
    args = parser.parse_args()

    if not os.path.exists(args.pdf_path):
        print(f"Error: PDF not found at {args.pdf_path}")
        return

    try:
        tokenizer, model, ocr = load_models()
        results = process_document(args.pdf_path, tokenizer, model, ocr)

        print("\n--- Extracted Values ---")
        if results:
            for key, val in results.items():
                print(f"{key}: {val}")
        else:
            print("No key information extracted.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
