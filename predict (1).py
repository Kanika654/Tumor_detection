import os
import torch
import h5py
import numpy as np
from torchvision import transforms
from models import CNNClassifier

def preprocess_image(h5_path, img_size=64):
    """Applies the same preprocessing steps as the training dataset."""
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"File not found: {h5_path}")
        
    with h5py.File(h5_path, 'r') as f:
        # Default keys usually 'image'
        key = list(f.keys())[0] if 'image' not in f.keys() else 'image'
        img = np.array(f[key])
        
    # Channel adjustment
    if len(img.shape) == 3 and img.shape[-1] <= 4:
        img = np.transpose(img, (2, 0, 1)) # to channels first
    elif len(img.shape) == 2:
        img = np.expand_dims(img, axis=0)
        
    tensor = torch.tensor(img, dtype=torch.float32)
    
    # Resize
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size), antialias=True),
    ])
    tensor = transform(tensor)
    
    # Normalize
    t_max = tensor.max()
    t_min = tensor.min()
    if t_max > t_min:
        tensor = (tensor - t_min) / (t_max - t_min)
    tensor = tensor * 2.0 - 1.0 # Scale to [-1, 1]
    
    # Add batch dimension
    tensor = tensor.unsqueeze(0)
    
    return tensor

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = r'd:\genai\balanced_cnn.pth'
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}. Please run train.py first.")
        return

    # User input
    print("====================================")
    print("🩺 Medical Image Predictor")
    print("====================================")
    file_path = input("Enter the absolute path to the .h5 image file: ").strip()
    
    # Preprocess
    try:
        print("\nLoading and preprocessing image...")
        input_tensor = preprocess_image(file_path, img_size=64)
        img_channels = input_tensor.shape[1]
    except Exception as e:
        print(f"Error processing image: {e}")
        return
        
    # Load Model
    print("Loading the trained model...")
    model = CNNClassifier(img_channels=img_channels, img_size=64)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    
    # Predict
    print("Making prediction...")
    with torch.no_grad():
        input_tensor = input_tensor.to(device)
        prob = model(input_tensor).item()
        
    is_tumor = prob > 0.5
    confidence = prob * 100 if is_tumor else (1 - prob) * 100
    
    print("\n====================================")
    print("📊 PREDICTION RESULTS")
    print("====================================")
    if is_tumor:
        print(f"Prediction: TUMOR DETECTED ⚠️")
    else:
        print(f"Prediction: NON-TUMOR (Healthy) ✅")
        
    print(f"Confidence: {confidence:.2f}%")
    print("====================================")

if __name__ == "__main__":
    main()
