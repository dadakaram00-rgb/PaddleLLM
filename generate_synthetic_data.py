import os
import json
import random
from PIL import Image, ImageDraw, ImageFont

# Define entities for German contracts
ENTITIES = {
    "TITLE": ["Mietvertrag", "Arbeitsvertrag", "Kaufvertrag", "Dienstleistungsvertrag"],
    "PARTY": ["Mustermann GmbH", "Schmidt & Co.", "Max Mustermann", "Erika Musterfrau"],
    "DATE": ["01.01.2023", "15.05.2024", "10.12.2022", "30.06.2023"],
    "AMOUNT": ["1000,00 Euro", "500 Euro", "1.250,50 €", "200,00 €"]
}

def generate_contract_text():
    title = random.choice(ENTITIES["TITLE"])
    party1 = random.choice(ENTITIES["PARTY"])
    party2 = random.choice(ENTITIES["PARTY"])
    date = random.choice(ENTITIES["DATE"])
    amount = random.choice(ENTITIES["AMOUNT"])

    lines = [
        (f"VERTRAG: {title}", "B-TITLE", "I-TITLE"),
        (f"Datum: {date}", "O", "B-DATE"),
        (f"Zwischen {party1}", "O", "B-PARTY"),
        (f"Und {party2}", "O", "B-PARTY"),
        (f"Gesamtbetrag: {amount}", "O", "B-AMOUNT")
    ]
    return lines

def create_synthetic_image(lines, output_path):
    width, height = 800, 600
    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)

    # Simple default font
    # Note: In a real environment, you might need to provide a path to a .ttf file
    font = ImageFont.load_default()

    words_data = []
    y_offset = 50

    for line_text, label_prefix, label_entity in lines:
        x_offset = 50
        words = line_text.split()
        for i, word in enumerate(words):
            # Estimate word size
            w_size = len(word) * 8
            h_size = 15

            bbox = [x_offset, y_offset, x_offset + w_size, y_offset + h_size]
            draw.text((x_offset, y_offset), word, fill=(0, 0, 0), font=font)

            # Determine label
            if i == 0 and label_prefix == "O":
                label = "O"
            elif i == 0 and label_prefix != "O":
                 label = label_prefix
            else:
                 label = label_entity if label_prefix != "O" or i > 0 else "O"

            # Fix labeling logic for simplicity in demo
            if "VERTRAG:" in line_text and i > 0: label = "I-TITLE"
            elif "VERTRAG:" in line_text and i == 1: label = "B-TITLE"
            elif date_in_word(word, ENTITIES["DATE"]): label = "B-DATE"
            elif any(p in line_text and word in p for p in ENTITIES["PARTY"]): label = "B-PARTY"
            elif any(a in line_text and word in a for a in ENTITIES["AMOUNT"]): label = "B-AMOUNT"
            else: label = "O"

            words_data.append({
                "word": word,
                "bbox": bbox,
                "label": label
            })
            x_offset += w_size + 10
        y_offset += 30

    image.save(output_path)
    return words_data, width, height

def date_in_word(word, dates):
    return any(d in word or word in d for d in dates)

def normalize_bbox(bbox, width, height):
    return [
        int(1000 * (bbox[0] / width)),
        int(1000 * (bbox[1] / height)),
        int(1000 * (bbox[2] / width)),
        int(1000 * (bbox[3] / height)),
    ]

def main(num_samples=5):
    dataset_dir = "synthetic_dataset"
    images_dir = os.path.join(dataset_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    train_data = []

    for i in range(num_samples):
        img_name = f"contract_{i}.jpg"
        img_path = os.path.join(images_dir, img_name)
        lines = generate_contract_text()
        words_data, w, h = create_synthetic_image(lines, img_path)

        sample = {
            "id": f"sample_{i}",
            "words": [wd["word"] for wd in words_data],
            "bbox": [normalize_bbox(wd["bbox"], w, h) for wd in words_data],
            "labels": [wd["label"] for wd in words_data],
            "image_path": img_path
        }
        train_data.append(sample)

    with open(os.path.join(dataset_dir, "train.jsonl"), "w") as f:
        for item in train_data:
            f.write(json.dumps(item) + "\n")

    print(f"Generated {num_samples} synthetic samples in {dataset_dir}")

if __name__ == "__main__":
    main()
