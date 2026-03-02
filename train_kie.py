import os
import json
import paddle
from paddlenlp.transformers import ErnieLayoutForTokenClassification, ErnieLayoutTokenizer
from paddle.io import Dataset, DataLoader
from PIL import Image
import numpy as np

# OFFLINE PATHS: Ensure these exist on your offline PC
MODEL_NAME = "./models/base_models/ernie-layoutx-base-uncased"
OUTPUT_DIR = "./models/fine_tuned/ernie_layout_kie"
DATASET_PATH = "./synthetic_dataset/train.jsonl"

# Training parameters (Optimized for 5GB RAM)
BATCH_SIZE = 1
LEARNING_RATE = 2e-5 # Smaller LR for stability on CPU
EPOCHS = 3

class ContractDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer):
        self.tokenizer = tokenizer
        self.data = []
        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(f"Dataset not found at {jsonl_path}. Please run generate_synthetic_data.py first.")

        with open(jsonl_path, "r") as f:
            for line in f:
                self.data.append(json.loads(line))

        # Including I- tags in label map
        self.label_map = {
            "O": 0,
            "B-TITLE": 1, "I-TITLE": 2,
            "B-DATE": 3, "I-DATE": 3,
            "B-PARTY": 4, "I-PARTY": 4,
            "B-AMOUNT": 5, "I-AMOUNT": 5
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # ErnieLayout requires image data
        image = Image.open(item["image_path"]).convert("RGB")
        image = image.resize((224, 224))
        image = np.array(image).transpose(2, 0, 1).astype("float32") / 255.0

        # Tokenization for ErnieLayout
        # Note: We manually align labels since word_ids is missing in some PaddleNLP tokenizers
        encoded_inputs = self.tokenizer(
            item["words"],
            bbox=item["bbox"],
            padding="max_length",
            truncation=True,
            max_seq_len=512,
            return_tensors="pd"
        )

        # Align labels manually
        # Standard approach: Tokenize each word individually to build a mapping
        labels = item["labels"]
        aligned_labels = [0] * 512 # Default to 'O' or ignore_index

        # Start after [CLS] token
        current_token_idx = 1
        for i, word in enumerate(item["words"]):
            # Tokenize word individually to see how many sub-tokens it creates
            word_tokens = self.tokenizer.tokenize(word)
            word_label = self.label_map.get(labels[i], 0)

            for _ in range(len(word_tokens)):
                if current_token_idx < 511: # Save space for [SEP]
                    aligned_labels[current_token_idx] = word_label
                    current_token_idx += 1

        # Set special tokens to ignore index
        aligned_labels[0] = -100 # [CLS]
        for j in range(current_token_idx, 512):
            aligned_labels[j] = -100 # [SEP] and [PAD]

        return {
            "input_ids": encoded_inputs["input_ids"][0],
            "bbox": encoded_inputs["bbox"][0],
            "image": image,
            "labels": paddle.to_tensor(aligned_labels, dtype="int64")
        }

def train():
    paddle.set_device("cpu")
    print(f"Loading local base model from: {MODEL_NAME}")

    if not os.path.exists(MODEL_NAME):
        print(f"ERROR: Model not found at {MODEL_NAME}. Please run download_models.py on an online machine first.")
        return

    tokenizer = ErnieLayoutTokenizer.from_pretrained(MODEL_NAME)
    model = ErnieLayoutForTokenClassification.from_pretrained(MODEL_NAME, num_labels=6)

    dataset = ContractDataset(DATASET_PATH, tokenizer)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Use SGD to save memory (AdamW uses 2x more state memory)
    # This helps stay within the 5GB limit on CPU
    optimizer = paddle.optimizer.SGD(learning_rate=LEARNING_RATE, parameters=model.parameters())
    criterion = paddle.nn.CrossEntropyLoss(ignore_index=-100)

    model.train()
    print("Starting fine-tuning on CPU (Offline Mode with memory optimization)...")
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch in loader:
            input_ids = batch["input_ids"]
            bbox = batch["bbox"]
            image = batch["image"]
            labels = batch["labels"]

            outputs = model(input_ids=input_ids, bbox=bbox, image=image)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs

            loss = criterion(logits.reshape([-1, 6]), labels.reshape([-1]))
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()

            total_loss += loss.numpy()[0]

        print(f"Epoch {epoch+1}/{EPOCHS}, Avg Loss: {total_loss/len(loader):.4f}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Fine-tuned model saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    train()
