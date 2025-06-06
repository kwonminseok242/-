import os
import numpy as np
import torch
import torch.nn as nn
from model import TCN

# 디바이스 설정 (GPU 사용 가능 시 활용)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===================================
# 검증 데이터 로드
# ===================================
current_dir = os.path.dirname(os.path.abspath(__file__))
validation_dir = os.path.join(current_dir, "..", "results")
X_val_path = os.path.join(validation_dir, "VX_windows_predict.npy")
y_val_path = os.path.join(validation_dir, "Vy_states_pridict.npy")

X_val = torch.tensor(np.load(X_val_path), dtype=torch.float32).to(device)
y_val = torch.tensor(np.load(y_val_path), dtype=torch.float32).to(device)

n_samples, window_size, n_features = X_val.shape

# ===================================
# 모델 정의 및 로드
# ===================================
model = TCN(input_size=n_features, output_size=1, num_channels=[32, 64, 64, 64], kernel_size=7).to(device)

model_save_path = os.path.join(current_dir, "..", "results", "models", "tcn_state_predictor.pth")
model.load_state_dict(torch.load(model_save_path, map_location=device))
model.eval()

# ===================================
# 평가
# ===================================
criterion_mse = nn.MSELoss()
criterion_mae = nn.L1Loss()

with torch.no_grad():
    # ❌ permute 제거 (모델이 내부 처리함)
    outputs = model(X_val)

    print("Output shape:", outputs.shape)
    print("Target shape:", y_val.shape)

    mse = criterion_mse(outputs, y_val).item()
    mae = criterion_mae(outputs, y_val).item()

print(f"[📝] Validation MSE: {mse:.4f}")
print(f"[📝] Validation MAE: {mae:.4f}")
