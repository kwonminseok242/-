import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import joblib

# --------------------------------------------------
# [1] 경로 설정
# --------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))          # 현재 파일 위치 (.../src)
project_root = os.path.dirname(current_dir)                       # 상위 폴더 (.../predict_final_final_final)
data_dir = os.path.join(project_root, "data")
results_dir = os.path.join(project_root, "results")

# 입력 및 출력 경로
train_path = os.path.join(data_dir, "T_f_t_d.csv")
predict_path = os.path.join(data_dir, "sensor_predictions_only.csv")
output_path = os.path.join(data_dir, "sensor_predictions_with_state.csv")
model_path = os.path.join(results_dir, "logistic_state_classifier.pkl")  # ✅ 모델 저장 위치 변경됨

# --------------------------------------------------
# [2] 데이터 로딩
# --------------------------------------------------
df_train = pd.read_csv(train_path)
df_predict = pd.read_csv(predict_path)

# 사용할 특성 및 라벨
feature_cols = ["PM2.5", "NTC", "CT1", "CT2", "CT3", "CT4", "temp_max"]
label_col = "state"

# 학습용 state 컬럼 확인
if label_col not in df_train.columns:
    raise ValueError("❌ 학습 데이터에 'state' 컬럼이 없습니다.")

X_train = df_train[feature_cols]
y_train = df_train[label_col]

# --------------------------------------------------
# [3] 로지스틱 회귀 모델 학습
# --------------------------------------------------
model = make_pipeline(
    StandardScaler(),
    LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
)
model.fit(X_train, y_train)

# 모델 저장
joblib.dump(model, model_path)
print(f"✅ 학습된 모델 저장 완료: {model_path}")

# --------------------------------------------------
# [4] 예측 수행
# --------------------------------------------------
X_pred = df_predict[feature_cols]
predicted_states = model.predict(X_pred)

# 결과 저장
df_predict["state"] = predicted_states
df_predict.to_csv(output_path, index=False)
print(f"📊 예측된 state 저장 완료: {output_path}")
