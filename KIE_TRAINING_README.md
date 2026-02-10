# Training a KIE Model with PaddleOCR

This guide explains how to train a Key Information Extraction (KIE) model using PaddleOCR's LayoutXLM algorithm on the provided synthetic German contract dataset.

## Prerequisites

1.  **Clone PaddleOCR Repository**:
    The training scripts are part of the PaddleOCR source code, not the pip package.
    ```bash
    git clone https://github.com/PaddlePaddle/PaddleOCR.git
    cd PaddleOCR
    ```

2.  **Install Dependencies**:
    Make sure you have `paddlepaddle`, `paddleocr`, and other requirements installed.
    ```bash
    pip install -r requirements.txt
    pip install -r PaddleOCR/requirements.txt
    ```

3.  **Prepare Backbone Model (Offline Usage)**:
    LayoutXLM requires a pre-trained backbone. If your machine is offline, you must download it on an online machine first.
    - Model: `layoutxlm-base-uncased` or `vi-layoutxlm-base-uncased`.
    - Usually, PaddleNLP handles this automatically. If offline, download the model files (`model_state.pdparams`, `tokenizer_config.json`, `vocab.txt`, etc.) from HuggingFace or PaddleNLP model zoo and place them in `~/.paddlenlp/models/layoutxlm-base-uncased/`.

## 1. Dataset Preparation

We have provided a script `generate_kie_dataset.py` that generates synthetic German contract data in the format required by PaddleOCR (XFUND format).

1.  Run the generator:
    ```bash
    python generate_kie_dataset.py
    ```
    This creates a `train_data/` directory containing:
    - `images/`: The generated contract images.
    - `train.txt`: Training annotations.
    - `test.txt`: Validation/Test annotations.
    - `class_list.txt`: List of entity classes (`OTHER`, `TITLE`, `DATE`, `PARTY`, `AMOUNT`).

2.  **Move Data to PaddleOCR Directory**:
    For convenience, move or symlink the `train_data` folder into the `PaddleOCR` directory.
    ```bash
    cp -r train_data PaddleOCR/train_data
    ```

## 2. Configuration

We provided a configuration file `kie_config.yml` tailored for this dataset.

1.  Copy the config to PaddleOCR:
    ```bash
    cp kie_config.yml PaddleOCR/configs/kie/vi_layoutxlm/kie_custom.yml
    ```
    *Note: You may need to create the directory `PaddleOCR/configs/kie/vi_layoutxlm/` if it doesn't exist, or place it in `PaddleOCR/configs/kie/`.*

2.  **Review Config**:
    Open `kie_custom.yml` and verify:
    - `data_dir`: Points to `./train_data`.
    - `label_file_list`: Points to `./train_data/train.txt`.
    - `class_path`: Points to `./train_data/class_list.txt`.
    - `num_classes`: 9 (Correct for 4 entities using BIO scheme).

## 3. Training

Run the training script from within the `PaddleOCR` directory.

```bash
cd PaddleOCR

# Run training (CPU)
python tools/train.py -c configs/kie/vi_layoutxlm/kie_custom.yml -o Global.use_gpu=False

# Run training (GPU)
# python tools/train.py -c configs/kie/vi_layoutxlm/kie_custom.yml -o Global.use_gpu=True
```

The training logs will appear, and checkpoints will be saved in `./output/kie_ser_layoutxlm/`.

## 4. Evaluation

Evaluate the trained model using the test set.

```bash
python tools/eval.py -c configs/kie/vi_layoutxlm/kie_custom.yml -o Global.checkpoints=./output/kie_ser_layoutxlm/best_accuracy
```

## 5. Model Export

To use the model for inference, you must export it to an inference model.

```bash
python tools/export_model.py -c configs/kie/vi_layoutxlm/kie_custom.yml -o Global.checkpoints=./output/kie_ser_layoutxlm/best_accuracy Global.save_inference_dir=./inference/kie_ser
```

This will create `inference/kie_ser/` containing `inference.pdmodel` and `inference.pdiparams`.

## 6. Inference

We provided a script `infer_kie.py` in the root of the project (outside PaddleOCR) to test the exported model.

1.  Ensure you are back in the project root.
2.  Run inference on a generated image:

```bash
python infer_kie.py --image_path train_data/images/contract_0.jpg --kie_model_dir PaddleOCR/inference/kie_ser
```

This will output the extracted entities and save a visualization in `output/inference_results/`.

## Troubleshooting

-   **OOM Errors**: Reduce `batch_size_per_card` in the config file.
-   **Missing Backbone**: If PaddleNLP tries to download `layoutxlm-base-uncased` and fails, manually download it and place it in the cache directory.
