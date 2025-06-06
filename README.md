semiconductor_tcn_project
|
|–– data                            # 센서 데이터 (원본/전처리)
|   |–– T_data.csv        # 기존 데이터
|   |–– T_f_data.csv                    # 상관관계 분석후 feature 제거 데이터
|   |–– T_f_t.csv                       # collection_date + collection_time = datetime merged 코드
|   |–– T_f_t_d.csv
|   |–– V_data.csv        # 기존 데이터
|   |–– V_f_data.csv                    # 상관관계 분석후 feature 제거 데이터
|   |–– V_f_t.csv                       # collection_date + collection_time = datetime merged 코드
|   |–– V_f_t_d.csv
|   |–– sensor_predictions_only.csv     # csv 예측된 센서 값만
|   |–– sensor_predictions.xlsx         # 예측된 센서 - 실제 센서값
|
|–– notebooks                       # EDA, 모델링 노트북
|   |–– T_data_preprovess.ipynb           # 데이터 전처리(상관관계분석,,,등)
|   |–– V_data_preprovess.ipynb          
|
|–– results                         # 결과 이미지, 모델 저장
|   |–– tcn_model.pth
|   |–– X_windows.npy  
|   |–– y_labels.npy
|   |–– VX_windows.npy  
|   |–– Vy_labels.npy
|   |––--------------- model
|                       |–– tcn_state_predictor.pth   
|–– src
|   |–– model.py                        # TCN 구현
|   |–– state_classifier_gui.py
|   |–– state_classifier_logistic.py
|   |–– state_gui_simulator.py          # 분류되는  gui (대시보드 기능)
|   |–– T_predict_evaluate.py                     # 테스트/시각화
|   |–– T_predict_s_window.py
|   |–– T_predict_train.py
|   |–– tcn_predict_demo.py    
|   |–– V_predict_s_window.py
|
|–– readme


#필요한 패키지
pip install numpy pandas matplotlib scikit-learn torch