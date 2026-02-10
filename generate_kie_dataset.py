import os
import random
import json
import shutil
from PIL import Image, ImageDraw, ImageFont

# Configuration
OUTPUT_DIR = "train_data"
IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")
TRAIN_LABEL_FILE = os.path.join(OUTPUT_DIR, "train.txt")
TEST_LABEL_FILE = os.path.join(OUTPUT_DIR, "test.txt")
LABEL_FILE = os.path.join(OUTPUT_DIR, "class_list.txt")
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
NUM_TRAIN = 20
NUM_TEST = 5

CLASSES = ["OTHER", "TITLE", "DATE", "PARTY", "AMOUNT"]

def create_dirs():
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(IMAGES_DIR)

def random_date():
    day = random.randint(1, 28)
    month = random.randint(1, 12)
    year = random.randint(2020, 2025)
    return f"{day}.{month}.{year}"

def generate_contract(image_id, is_train=True):
    width, height = 800, 1100
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    try:
        font_large = ImageFont.truetype(FONT_PATH, 30)
        font_medium = ImageFont.truetype(FONT_PATH, 20)
        font_small = ImageFont.truetype(FONT_PATH, 14)
    except Exception as e:
        print(f"Warning: Could not load font {FONT_PATH}. Using default.")
        font_large = ImageFont.load_default()
        font_medium = ImageFont.load_default()
        font_small = ImageFont.load_default()

    annotations = []

    # Helper to add text and annotation
    def add_text(text, position, font, label="OTHER"):
        x, y = position
        bbox = draw.textbbox((x, y), text, font=font)
        draw.text((x, y), text, font=font, fill="black")

        # Convert bbox to points [x1, y1, x2, y2, x3, y3, x4, y4]
        # bbox is (left, top, right, bottom)
        left, top, right, bottom = bbox
        points = [[left, top], [right, top], [right, bottom], [left, bottom]]

        annotations.append({
            "transcription": text,
            "label": label,
            "points": points,
            "id": len(annotations),
            "linking": []
        })
        return bottom # Return bottom y for next line

    current_y = 50

    # 1. Title
    title_text = random.choice(["DIENSTLEISTUNGSVERTRAG", "MIETVERTRAG", "ARBEITSVERTRAG", "KAUFVERTRAG"])
    current_y = add_text(title_text, (50, current_y), font_large, "TITLE") + 40

    # 2. Date
    date_label = "Datum:"
    date_val = random_date()
    # Draw label
    add_text(date_label, (50, current_y), font_medium, "OTHER")
    # Draw value
    add_text(date_val, (150, current_y), font_medium, "DATE")
    current_y += 40

    # 3. Parties
    party1 = random.choice(["Alpha GmbH", "Beta AG", "Schmidt & Co.", "Müller Logistics"])
    party2 = random.choice(["Max Mustermann", "Erika Musterfrau", "Hans Müller", "Petra Schmidt"])

    add_text("Zwischen:", (50, current_y), font_medium, "OTHER")
    current_y += 30
    add_text(party1, (50, current_y), font_medium, "PARTY")
    current_y += 30
    add_text("und", (50, current_y), font_medium, "OTHER")
    current_y += 30
    add_text(party2, (50, current_y), font_medium, "PARTY")
    current_y += 50

    # 4. Content (Filler)
    filler_texts = [
        "Dieser Vertrag regelt die Zusammenarbeit zwischen den Parteien.",
        "Die Leistungen sind wie folgt vereinbart.",
        "Die Zahlung erfolgt innerhalb von 30 Tagen.",
        "Gerichtsstand ist Berlin."
    ]
    for _ in range(3):
        text = random.choice(filler_texts)
        current_y = add_text(text, (50, current_y), font_small, "OTHER") + 10

    current_y += 30

    # 5. Amount
    amount_label = "Gesamtbetrag:"
    amount_val = f"{random.randint(100, 10000)},00 EUR"

    add_text(amount_label, (50, current_y), font_medium, "OTHER")
    add_text(amount_val, (250, current_y), font_medium, "AMOUNT")
    current_y += 60

    # Save Image
    image_filename = f"contract_{image_id}.jpg"
    image_path = os.path.join(IMAGES_DIR, image_filename)
    image.save(image_path)

    return image_filename, annotations

def generate_dataset():
    create_dirs()

    # Write Label List
    with open(LABEL_FILE, "w") as f:
        for cls in CLASSES:
            f.write(cls + "\n")

    # Generate Train
    train_data = []
    print("Generating Training Data...")
    for i in range(NUM_TRAIN):
        fname, anns = generate_contract(i, is_train=True)
        # PaddleOCR format in json/txt: usually per line: image_path\tjson_dump
        rel_path = f"images/{fname}"
        train_data.append(f"{rel_path}\t{json.dumps(anns)}")

    with open(TRAIN_LABEL_FILE, "w", encoding="utf-8") as f:
        for line in train_data:
            f.write(line + "\n")

    # Generate Test
    test_data = []
    print("Generating Test Data...")
    for i in range(NUM_TEST):
        fname, anns = generate_contract(NUM_TRAIN + i, is_train=False)
        rel_path = f"images/{fname}"
        test_data.append(f"{rel_path}\t{json.dumps(anns)}")

    with open(TEST_LABEL_FILE, "w", encoding="utf-8") as f:
        for line in test_data:
            f.write(line + "\n")

    print(f"Generated {NUM_TRAIN} training samples and {NUM_TEST} test samples in {OUTPUT_DIR}")

if __name__ == "__main__":
    generate_dataset()
