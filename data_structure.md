# Data Structure for LayoutXLM Fine-tuning

To fine-tune LayoutXLM for Key Information Extraction (KIE), the data should be structured in a way that includes both textual content and spatial information (bounding boxes).

## JSON Format (JSON Lines)

The recommended format is a JSON Lines (`.jsonl`) file where each line represents one document page.

### Example Entry:

```json
{
  "id": "doc_001",
  "words": ["Vertrag", "Datum:", "01.01.2024", "Betrag:", "100", "Euro"],
  "bbox": [
    [100, 100, 200, 120],
    [100, 130, 150, 150],
    [160, 130, 250, 150],
    [100, 160, 150, 180],
    [160, 160, 200, 180],
    [210, 160, 250, 180]
  ],
  "labels": ["B-TITLE", "O", "B-DATE", "O", "B-AMOUNT", "I-AMOUNT"],
  "image_path": "images/doc_001.jpg"
}
```

### Field Definitions:

1.  **`words`** (List[str]): The individual tokens extracted from the document via OCR.
2.  **`bbox`** (List[List[int]]): The bounding box for each word.
    - Format: `[x1, y1, x2, y2]`
    - Coordinates should be normalized to a scale of 0-1000 relative to the image size.
3.  **`labels`** (List[str]): The BIO (Begin, Inside, Outside) tags for each word.
    - Example: `B-TITLE`, `I-TITLE`, `B-DATE`, `B-PARTY`, `B-AMOUNT`, `O`.
4.  **`image_path`** (str): Path to the original image file. LayoutXLM uses the image for visual feature extraction.

## Normalizing Bounding Boxes

Bounding boxes must be normalized to a 1000x1000 scale:

```python
def normalize_bbox(bbox, width, height):
    return [
        int(1000 * (bbox[0] / width)),
        int(1000 * (bbox[1] / height)),
        int(1000 * (bbox[2] / width)),
        int(1000 * (bbox[3] / height)),
    ]
```

## Dataset Directory Structure

```text
dataset/
├── images/
│   ├── doc_001.jpg
│   ├── doc_002.jpg
│   └── ...
├── train.jsonl
└── val.jsonl
```
