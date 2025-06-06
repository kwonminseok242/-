import os
import torch
import numpy as np
import pandas as pd
from model import TCN

# --------------------------------------------------
# [1] 경로 설정
# --------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))          # 현재 파일 위치 (.../src)
project_root = os.path.dirname(current_dir)                       # 최상위 폴더 (.../predict_final_final_final)
results_dir = os.path.join(project_root, "results")               # 결과 폴더
model_dir = os.path.join(results_dir, "models")                   # 모델 저장 폴더 (수정됨)
data_dir = os.path.join(project_root, "data")                     # 예측 결과 저장 위치

# 경로 지정
X_path = os.path.join(results_dir, "X_windows_predict.npy")
y_path = os.path.join(results_dir, "y_states_pridict.npy")
model_path = os.path.join(model_dir, "tcn_state_predictor.pth")
csv_path = os.path.join(data_dir, "sensor_predictions_only.csv")  # 저장될 CSV 경로

# 🔍 경로 존재 여부 확인
print(f"📁 모델 경로: {model_path}")
if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ 모델 파일이 존재하지 않습니다: {model_path}")

# --------------------------------------------------
# [2] 데이터 로딩
# --------------------------------------------------
X = np.load(X_path)
y = np.load(y_path)
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

n_samples, window_size, n_features = X.shape
output_size = y.shape[1]
print("✅ Loaded data:", f"Samples={n_samples}, Features={n_features}, Output size={output_size}")

# --------------------------------------------------
# [3] 모델 초기화 및 로딩
# --------------------------------------------------
model = TCN(input_size=n_features,
            output_size=output_size,
            num_channels=[32, 64, 64, 64],
            kernel_size=7)

model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()

# --------------------------------------------------
# [4] 예측값만 저장
# --------------------------------------------------
column_names = ["PM2.5", "NTC", "CT1", "CT2", "CT3", "CT4", "temp_max"]
predictions = []

with torch.no_grad():
    for i in range(1000):  # 원하는 샘플 수만큼 조절
        input_seq = X_tensor[i].unsqueeze(0)           # [1, 50, 센서수]
        prediction = model(input_seq)[0].numpy()       # [센서수]
        predictions.append(prediction)

        print(f"\n[Sample {i + 1}]")
        print("🔹 예측값 :", np.round(prediction, 3))

# --------------------------------------------------
# [5] 예측값 CSV로 저장
# --------------------------------------------------
df = pd.DataFrame(predictions, columns=column_names)
df.to_csv(csv_path, index=False)
print(f"\n📊 예측값만 저장되었습니다: {csv_path}")
