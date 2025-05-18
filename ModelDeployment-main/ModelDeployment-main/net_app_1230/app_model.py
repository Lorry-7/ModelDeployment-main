import streamlit as st  # 导入 Streamlit 库，用于创建 Web 应用
import pandas as pd  # 导入 Pandas 库，用于数据处理
import pickle  # 导入 pickle 库，用于加载已训练的模型
import os  # 导入 os 库，用于处理文件路径
from sklearn.metrics import accuracy_score

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'stacking_clf.pkl')

with open(model_path, 'rb') as file:
    model = pickle.load(file)

st.title("Predicting liver metastasis in gastric cancer using Stacking machine learning model")

st.sidebar.header("Selection Panel")
st.sidebar.subheader("Picking up parameters")

Age_option = ["0-49 years", "50-59 years", "60-69 years", "70-79 years", "80+ years"]
Age_map = {"0-49 years": 0, "50-59 years": 1, "60-69 years": 2, "70-79 years": 3, "80+ years": 4}
Age_sb = st.sidebar.selectbox("Age", Age_option, index=4)

Sex_option = {"Female", "Male"}
Sex_map = {"Female": 0, "Male": 1}
Sex_sb = st.sidebar.selectbox("Sex", Sex_option, index=1)

Primary_site_option = {"Cardia", "Fundus of stomach", "Body of stomach", "Gastric antrum", "Pylorus", "Lesser curvature of stomach",
                       "Greater curvature of stomach", "Overlapping lesion of stomach", "Stomach"}
Primary_site_map = {"Cardia": 0, "Fundus of stomach": 1, "Body of stomach": 2, "Gastric antrum": 3, "Pylorus": 4,
                    "Lesser curvature of stomach": 5, "Greater curvature of stomach": 6,
                    "Overlapping lesion of stomach": 7, "Stomach": 8}
Primary_site_sb = st.sidebar.selectbox("Primary site", Primary_site_option, index=8)

Histological_type_option = {"Adenocarcinoma", "Signet ring cell carcinoma", "Spacial type"}
Histological_type_map = {"Adenocarcinoma": 0, "Signet ring cell carcinoma": 1, "Spacial type": 2}
Histological_type_sb = st.sidebar.selectbox("Histological type", Histological_type_option, index=0)

T_Stage_option = {"T1", "T2", "T3", "T4", "TX"}
T_Stage_map = {"T1": 0, "T2": 1, "T3": 2, "T4": 3, "TX": 4}
T_Stage_sb = st.sidebar.selectbox("T Stage", T_Stage_option, index=4)

N_Stage_option = {"N0", "N1", "N2", "N3", "NX"}
N_Stage_map = {"N0": 0, "N1": 1, "N2": 2, "N3": 3, "NX": 4}
N_Stage_sb = st.sidebar.selectbox("N Stage", N_Stage_option, index=0)

Surgery_status_option = {"No", "Surgery performed", "Unknown"}
Surgery_status_map = {"No": 0, "Surgery performed": 1, "Unknown": 2}
Surgery_status_sb = st.sidebar.selectbox("Surgery status", Surgery_status_option, index=0)

Radiation_status_option = {"Radiation", "Refused", "Unknown"}
Radiation_status_map = {"Radiation": 0, "Refused": 1, "Unknown": 2}
Radiation_status_sb = st.sidebar.selectbox("Radiation status", Radiation_status_option, index=2)

Chemotherapy_status_option = {"No/Unknown", "Yes"}
Chemotherapy_status_map = {"No/Unknown": 0, "Yes": 1}
Chemotherapy_status_sb = st.sidebar.selectbox("Chemotherapy status", Chemotherapy_status_option, index=1)

Extrahepatic_metastasis_option = {"No", "Unknown", "Yes"}
Extrahepatic_metastasis_map = {"No": 0, "Unknown": 1, "Yes": 2}
Extrahepatic_metastasis_sb = st.sidebar.selectbox("Extrahepatic metastasis", Extrahepatic_metastasis_option, index=0)

Tumor_size_option = {"≥2cm but <5cm", "≥5cm", "Unknown", "<2cm"}
Tumor_size_map = {"≥2cm but <5cm": 0, "≥5cm": 1, "Unknown": 2, "<2cm": 3}
Tumor_size_sb = st.sidebar.selectbox("Tumor size", Tumor_size_option, index=1)


input_data = pd.DataFrame({
    'Age': [Age_map[Age_sb]],
    'Sex': [Sex_map[Sex_sb]],
    'Primary site': [Primary_site_map[Primary_site_sb]],
    'Histological type': [Histological_type_map[Histological_type_sb]],
    'T Stage': [T_Stage_map[T_Stage_sb]],
    'N Stage': [N_Stage_map[N_Stage_sb]],
    'Surgery status': [Surgery_status_map[Surgery_status_sb]],
    'Radiation status': [Radiation_status_map[Radiation_status_sb]],
    'Chemotherapy status': [Chemotherapy_status_map[Chemotherapy_status_sb]],
    'Extrahepatic metastasis': [Extrahepatic_metastasis_map[Extrahepatic_metastasis_sb]],
    'Tumor size': [Tumor_size_map[Tumor_size_sb]]
})

if st.button("Predict"):
    y_pred = model.predict(input_data)
    y_pred_proba = model.predict_proba(input_data)[:, 1]
    final_pred_proba = y_pred_proba[0] * 100
    st.write(f"Predicted probability of liver metastasis: {final_pred_proba:.2f}%")

