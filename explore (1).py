import pandas as pd
import h5py
import os

meta_path = r'd:\genai\genai\BraTS2020_training_data\content\data\meta_data.csv'
base_dir = r'd:\genai\genai\BraTS2020_training_data'

df = pd.read_csv(meta_path)
print("Distribution:")
print(df['target'].value_counts())

# Check first file
first_file_rel = df['slice_path'].iloc[0].lstrip('/') # e.g. content/data/volume_41_slice_0.h5
first_file_abs = os.path.join(base_dir, first_file_rel)
print(f"Path: {first_file_abs}, Exists: {os.path.exists(first_file_abs)}")

if os.path.exists(first_file_abs):
    with h5py.File(first_file_abs, 'r') as f:
        print("Keys:", list(f.keys()))
        for k in f.keys():
            print(f"{k} shape:", f[k].shape)
