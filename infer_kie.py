import os
import cv2
import paddle
from paddleocr import PPStructure, save_structure_res
import argparse

def main():
    parser = argparse.ArgumentParser(description="Inference with trained KIE model")
    parser.add_argument("--image_path", type=str, required=True, help="Path to image")
    parser.add_argument("--kie_model_dir", type=str, default="./output/kie_ser_layoutxlm/best_accuracy", help="Path to exported inference model")
    parser.add_argument("--vis_font_path", type=str, default="/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", help="Font path for visualization")
    args = parser.parse_args()

    # Ensure CPU
    paddle.set_device("cpu")

    # Initialize KIE engine
    # PPStructure supports 'kie' mode.
    # We need to pass the model path.
    # Note: PPStructure usually expects inference models (exported models), not checkpoint models.
    # The training output usually needs to be exported first using export_model.py in PaddleOCR.
    # Here we assume the user has exported the model to 'kie_model_dir'.

    # We also need ser_dict_path which is the class list.
    class_path = "./train_data/class_list.txt"

    print(f"Loading KIE model from {args.kie_model_dir}...")

    try:
        # Initialize the KIE engine
        # mode='kie' enables Key Information Extraction
        # kie_algorithm='LayoutXLM' specifies the algorithm
        engine = PPStructure(
            recovery=False,
            structure_version='PP-StructureV2',
            mode='kie',
            kie_algorithm='LayoutXLM',
            ser_model_dir=args.kie_model_dir,
            ser_dict_path=class_path,
            use_gpu=False,
            show_log=True,
            image_orientation=False
        )
    except Exception as e:
        print(f"Error initializing PPStructure: {e}")
        print("Make sure you have trained and EXPORTED the model first.")
        print("Note: The model directory must contain 'inference.pdmodel', 'inference.pdiparams'.")
        return

    img_path = args.image_path
    if not os.path.exists(img_path):
        print(f"Image not found: {img_path}")
        return

    print(f"Processing {img_path}...")
    img = cv2.imread(img_path)
    if img is None:
        print("Failed to read image.")
        return

    # Run Inference
    try:
        result = engine(img)
    except Exception as e:
        print(f"Inference failed: {e}")
        return

    # Result is a list of dicts. For KIE, it usually contains 'ser_res'.
    # ser_res structure: [{'transcription': '...', 'bbox': [...], 'pred': 'LABEL', ...}]

    if result:
        print("\nExtraction Results:")
        # The result from PPStructure in KIE mode is usually a list (one per image region, usually 1 for full page)
        for idx, res in enumerate(result):
             print(f"--- Region {idx} ---")
             # Depending on version, it might be in different keys, but usually just the list itself if it's purely SER?
             # Or inside 'res'.
             # Let's inspect the keys if possible, but for now we assume standard PPStructure output.
             # If using SER only, result[0] is usually a list of entities.

             # PPStructure returns a list of elements. For KIE, it wraps SER results.
             if isinstance(res, dict):
                 if 'res' in res:
                     # This is expected for structure recovery, but for KIE mode it might differ.
                     # Let's print the keys available.
                     print(f"Keys: {res.keys()}")
                     if 'res' in res:
                         print(res['res'])
                 else:
                     print(res)
             else:
                 print(res)

        # Visualize
        save_folder = './output/inference_results'
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        try:
            save_structure_res(result, save_folder, os.path.basename(img_path), font_path=args.vis_font_path)
            print(f"\nVisualization saved to {save_folder}")
        except Exception as e:
            print(f"Visualization failed: {e}")
    else:
        print("No results found.")

if __name__ == "__main__":
    main()
