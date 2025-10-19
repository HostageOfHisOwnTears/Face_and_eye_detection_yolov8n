import pandas as pd

# Specify the path to the CSV file containing training/validation metrics
csv_path = "tested-runs/detect/face-eye-detector/results.csv"  

# Load the CSV file containing training/validation metrics
df = pd.read_csv(csv_path)

# Retrieve the last row of the CSV, which corresponds to the final epoch
last_epoch = df.iloc[-1]

# Display the validation metrics for the final epoch
print("Validation metrics for the last epoch:")
print(f"Epoch: {int(last_epoch['epoch'])}")  # Display the epoch number
print(f"Precision: {last_epoch['metrics/precision(B)']:.4f}")  # Precision score for class B
print(f"Recall: {last_epoch['metrics/recall(B)']:.4f}")        # Recall score for class B
print(f"mAP@0.5: {last_epoch['metrics/mAP50(B)']:.4f}")        # Mean Average Precision at IoU=0.5
print(f"mAP@0.5:0.95: {last_epoch['metrics/mAP50-95(B)']:.4f}") # Mean Average Precision at IoU=0.5:0.95
