import numpy as np
import pandas as pd
import os
from PIL import Image
from ce_utils import (
    load_single_image,
    load_model,
    predict_labels,
    write_inference_excel,
    standardize_transform
)
from mnist_logreg_idx import (
    SoftmaxLogisticRegression,
    preprocess_images
)


def ce_inference(data_path):
    model_path = "./celegans_model.npz"
    if not os.path.isdir(data_path):
        raise FileNotFoundError(f"Data folder not found: {data_path}")

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    w, b, image_shape, mean, std = load_model(model_path)

    files = sorted([f for f in os.listdir(data_path) if f.lower().endswith(".png")])
    if not files:
        raise ValueError("No .png files found in the provided folder.")

    X = []
    filenames = []

    for fname in files:
        full_path = os.path.join(data_path, fname)
        flat, shape = load_single_image(full_path, expected_shape=image_shape)
        X.append(flat)
        filenames.append(fname)

    X = np.array(X, dtype=np.float32)
    if mean is not None and std is not None:
        X = standardize_transform(X, mean, std)
    else:
        raise ValueError("Mean and std values are required for standardization but were not found in the model file.")
    preds = predict_labels(X, w, b, threshold=0.5)

    filename = "celegans_inference_output.xlsx"
    output_excel = os.path.join("./", filename)
    write_inference_excel(output_excel, filenames, preds.tolist())

    print(f"Inference complete.")
    print(f"Excel output written to: {filename}")
    print("Note: The label counts are in the 'Label Counts' sheet of the Excel file.")
    
    
def mnist_inference(data_path):
    model_path = "./mnist_softmax_model.npz"
    if not os.path.isdir(data_path):
        raise FileNotFoundError(f"Data folder not found: {data_path}")

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model = SoftmaxLogisticRegression.load(model_path)
    files = sorted([f for f in os.listdir(data_path) if f.lower().endswith(".tif")])
    if not files:
        raise ValueError("No .tif files found in the provided folder.")
    
    X = []
    filenames = []
    
    for fname in files:
        full_path = os.path.join(data_path, fname)
        img = Image.open(full_path).convert("L")  # grayscale
        img = img.resize((28, 28))  # Ensure image size is 28x28
        img = np.array(img, dtype=np.float32)
        if np.mean(img) > 127:
            img = 255 - img  # invert colors if background is white
        X.append(img)
        filenames.append(fname)
    
    X = np.array(X, dtype=np.float32)
    X = preprocess_images(X)
    preds = model.predict(X)
    
    output_excel = "mnist_inference_output.xlsx"
    df_pred = pd.DataFrame({'Filename': filenames, 'Predicted_Label': preds})
    label_counts = pd.Series(preds).value_counts().sort_index()
    df_count = pd.DataFrame({'Label': label_counts.index, 'Count': label_counts.values})
    with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
        df_pred.to_excel(writer, sheet_name='Predictions', index=False)
        df_count.to_excel(writer, sheet_name='Label_Counts', index=False)

    print("MNIST inference complete.")
    print(f"Excel output written to: {output_excel}")
    print("Note: The label counts are in the 'Label Counts' sheet of the Excel file.")

def main():
    print("Would you like to test the MNIST model or C. Elegans Model? Enter 'M' or 'CE':")
    model_choice = input().strip().upper()
    while model_choice not in ['M', 'CE']:
        print("Invalid choice. Please enter 'M' for MNIST or 'CE' for C. Elegans:")
        model_choice = input().strip().upper()
        
    print("Please enter the absolute path to the folder for inference data:")
    data_path = input().strip()
    
    if model_choice == 'M':
        mnist_inference(data_path)
    elif model_choice == 'CE':
        ce_inference(data_path)
            
if __name__ == "__main__":
    main()
    