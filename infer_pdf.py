import os
import argparse
import paddle
import pypdfium2 as pdfium
from PIL import Image
from paddleocr import PaddleOCR
from paddlenlp.transformers import LayoutXLMForTokenClassification, LayoutXLMTokenizer
import numpy as np

# Label mapping (same as training)
ID2LABEL = {0: "O", 1: "TITLE", 2: "TITLE", 3: "DATE", 4: "PARTY", 5: "AMOUNT"}

def load_models(model_path="models/layoutxlm_kie"):
    paddle.set_device("cpu")
    try:
        tokenizer = LayoutXLMTokenizer.from_pretrained(model_path)
        model = LayoutXLMForTokenClassification.from_pretrained(model_path)
    except:
        print(f"Custom model not found at {model_path}. Using base model for demo.")
        tokenizer = LayoutXLMTokenizer.from_pretrained("microsoft/layoutxlm-base")
        model = LayoutXLMForTokenClassification.from_pretrained("microsoft/layoutxlm-base", num_labels=6)

    model.eval()
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

    # OCR
    img_array = np.array(image)
    ocr_result = ocr.ocr(img_array, cls=True)

    words = []
    bboxes = []

    for line in ocr_result[0]:
        text = line[1][0]
        box = line[0]
        words.append(text)
        bboxes.append(normalize_bbox([box[0][0], box[0][1], box[2][0], box[2][1]], w, h))

    vision_image = image.resize((224, 224)).convert("RGB")
    vision_image = np.array(vision_image).transpose(2, 0, 1).astype("float32") / 255.0
    vision_image = paddle.to_tensor([vision_image])

    # Model Inference
    inputs = tokenizer(
        words,
        bbox=bboxes,
        return_tensors="pd"
    )

    # Extract word_ids to map predictions back to OCR words
    word_ids = inputs.word_ids()

    with paddle.no_grad():
        outputs = model(
            input_ids=inputs["input_ids"],
            bbox=inputs["bbox"],
            image=vision_image
        )

    logits = outputs[0] if isinstance(outputs, tuple) else outputs
    predictions = paddle.argmax(logits, axis=-1).numpy()[0]

    # Map token-level predictions back to words
    extracted_data = {}
    last_word_idx = None

    for pred_idx, word_idx in zip(predictions, word_ids):
        if word_idx is None or word_idx == last_word_idx:
            # Skip special tokens or subsequent sub-tokens of the same word
            continue

        last_word_idx = word_idx
        label = ID2LABEL.get(pred_idx, "O")
        if label != "O":
            if label not in extracted_data:
                extracted_data[label] = []
            extracted_data[label].append(words[word_idx])

    # Format the list of words into strings
    for label in extracted_data:
        extracted_data[label] = " ".join(extracted_data[label])

    return extracted_data

def main():
    parser = argparse.ArgumentParser(description="PDF KIE Inference using LayoutXLM")
    parser.add_argument("pdf_path", help="Path to the PDF document")
    args = parser.parse_args()

    tokenizer, model, ocr = load_models()
    results = process_document(args.pdf_path, tokenizer, model, ocr)

    print("\n--- Extracted Values ---")
    for key, val in results.items():
        print(f"{key}: {val}")

if __name__ == "__main__":
    main()
