import tkinter as tk
from tkinter import ttk
import pandas as pd
import joblib
import os
import time
import threading

# (1) 경로 설정: __file__ 대신 절대경로 혹은 현재 작업 디렉토리로 바꿔보기
project_root = os.getcwd()
data_path = os.path.join(project_root, "data", "sensor_predictions_only.csv")
model_path = os.path.join(project_root, "results", "logistic_state_classifier.pkl")

# (2) 데이터/모델 로딩
try:
    df = pd.read_csv(data_path)
    print("CSV 로딩 성공:", df.shape)
except Exception as e:
    print("CSV 로딩 실패:", e)
    df = pd.DataFrame()  # 빈 DataFrame

try:
    model = joblib.load(model_path)
    print("모델 로딩 성공:", model)
except Exception as e:
    print("모델 로딩 실패:", e)
    model = None

# (3) GUI 초기화
root = tk.Tk()
root.title("📊 실시간 센서 상태 분류 시뮬레이터")
root.geometry("1000x500")

style = ttk.Style()
style.theme_use("default")

# Treeview 생성
tree = ttk.Treeview(root, show="headings")
tree.pack(expand=True, fill=tk.BOTH)

# 컬럼명 설정 (csv가 비어 있으면 기본값 없음)
columns = list(df.columns) + ["Predicted State"]
tree["columns"] = columns
for col in columns:
    tree.heading(col, text=col)
    tree.column(col, width=100, anchor="center")

# 스크롤바
scrollbar = ttk.Scrollbar(root, orient="vertical", command=tree.yview)
tree.configure(yscroll=scrollbar.set)
scrollbar.pack(side="right", fill="y")

tree.tag_configure("state_0", background="#f0f0f0")
tree.tag_configure("state_1", background="#d0f5d0")
tree.tag_configure("state_2", background="#d0e0ff")
tree.tag_configure("state_3", background="#f8d0d0")

# (4) 시뮬레이션 함수 (에러 출력 추가)
def simulate_predictions():
    if df.empty or model is None:
        print("df가 비어있거나 model이 None이라 시뮬레이션을 할 수 없습니다.")
        return

    for idx, row in df.iterrows():
        try:
            features = row.values.reshape(1, -1)
            predicted_state = model.predict(features)[0]
            values = list(row.values) + [predicted_state]
            tag = f"state_{predicted_state}"
            tree.insert("", tk.END, values=values, tags=(tag,))
            tree.see(tree.get_children()[-1])
            time.sleep(1)
        except Exception as e:
            print(f"[Error] idx={idx} 에서 예외: {e}")
            break

# (5) 스레드로 실행
def start_simulation():
    start_btn.config(state="disabled")
    threading.Thread(target=simulate_predictions, daemon=True).start()

# (6) 시작 버튼
start_btn = ttk.Button(root, text="▶ 시뮬레이션 시작", command=start_simulation)
start_btn.pack(pady=10)

root.mainloop()
