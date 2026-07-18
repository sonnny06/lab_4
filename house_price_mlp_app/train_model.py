"""Optional training script for the MLP house price model.
Run: python train_model.py
"""
from model_utils import train_mlp

if __name__ == "__main__":
    bundle = train_mlp(epochs=100, batch_size=32, lr=0.001, seed=42)
    print("MLP Regression Results")
    print(f"MSE: {bundle['metrics']['MSE']:.6f}")
    print(f"MAE: {bundle['metrics']['MAE']:.6f}")
    print(f"R2 : {bundle['metrics']['R2']:.6f}")
