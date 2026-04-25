import os

base_dir = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(base_dir, "model", "mental_health_model.pkl")
cols_path = os.path.join(base_dir, "model", "columns.pkl")

print("======== DEBUG START ========")
print("BASE DIR:", base_dir)
print("FILES IN BASE DIR:", os.listdir(base_dir))

model_dir = os.path.join(base_dir, "model")
print("MODEL DIR EXISTS:", os.path.exists(model_dir))

if os.path.exists(model_dir):
    print("FILES IN MODEL DIR:", os.listdir(model_dir))

print("MODEL PATH:", model_path)
print("MODEL EXISTS:", os.path.exists(model_path))

print("COLS PATH:", cols_path)
print("COLS EXISTS:", os.path.exists(cols_path))
print("======== DEBUG END ========")