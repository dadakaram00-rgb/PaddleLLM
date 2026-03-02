import os
import json
import paddle
from paddlenlp.transformers import LayoutXLMForTokenClassification, LayoutXLMTokenizer
from paddle.io import Dataset, DataLoader
from PIL import Image
import numpy as np

# Memory-efficient training parameters for CPU (5GB limit)
BATCH_SIZE = 1
LEARNING_RATE = 5e-5
EPOCHS = 3
MODEL_NAME = "microsoft/layoutxlm-base"

class ContractDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer):
        self.tokenizer = tokenizer
        self.data = []
        with open(jsonl_path, "r") as f:
            for line in f:
                self.data.append(json.loads(line))

        self.label_map = {"O": 0, "B-TITLE": 1, "I-TITLE": 2, "B-DATE": 3, "B-PARTY": 4, "B-AMOUNT": 5}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.open(item["image_path"]).convert("RGB")
        image = image.resize((224, 224))
        image = np.array(image).transpose(2, 0, 1).astype("float32") / 255.0

        # Tokenize with alignment info
        inputs = self.tokenizer(
            item["words"],
            bbox=item["bbox"],
            padding="max_length",
            truncation=True,
            max_seq_len=512,
            return_attention_mask=True,
            return_tensors="pd"
        )

        # Align labels with sub-tokens
        # LayoutXLMTokenizer returns word_ids which map each token to its original word index
        word_ids = inputs.word_ids()
        labels = item["labels"]

        aligned_labels = []
        for word_idx in word_ids:
            if word_idx is None:
                # Special tokens like [CLS], [SEP], [PAD]
                aligned_labels.append(-100) # Standard ignore index for CrossEntropyLoss
            else:
                # Map the label of the original word to this sub-token
                label_str = labels[word_idx]
                aligned_labels.append(self.label_map.get(label_str, 0))

        return {
            "input_ids": inputs["input_ids"][0],
            "bbox": inputs["bbox"][0],
            "image": image,
            "labels": paddle.to_tensor(aligned_labels, dtype="int64")
        }

def train():
    paddle.set_device("cpu")
    tokenizer = LayoutXLMTokenizer.from_pretrained(MODEL_NAME)
    model = LayoutXLMForTokenClassification.from_pretrained(MODEL_NAME, num_labels=6)

    dataset = ContractDataset("synthetic_dataset/train.jsonl", tokenizer)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = paddle.optimizer.AdamW(learning_rate=LEARNING_RATE, parameters=model.parameters())
    # Use ignore_index to skip special tokens in loss calculation
    criterion = paddle.nn.CrossEntropyLoss(ignore_index=-100)

    model.train()
    print("Starting fine-tuning on CPU (with correct label alignment)...")
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

    output_dir = "models/layoutxlm_kie"
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")

if __name__ == "__main__":
    train()
