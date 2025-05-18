
import streamlit as st
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt
import os  # 导入 os 库，用于处理文件路径

# 加载模型
# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 组合当前目录与模型文件名，生成模型的完整路径
model_path = os.path.join(current_dir, 'MLPmodel.pkl')
# 打开并加载模型
with open(model_path, 'rb') as file:
    model = joblib.load(file)  # 使用 pickle 加载模型文件

# # 加载模型
# model = joblib.load('MLPmodel.pkl')

# 特征名称
feature_names = [
    "Age", "Symptom Duration", "Flexion", "Extension", "Rotation",
    "Spring test-pain", "Muscle tightness", "Exacerbation on Flexion",
    "Exacerbation on Extension"
]

# Streamlit 应用标题
st.title("A Quick Screening Tool for Predicting NP Patients Benefiting from SMT")   

# 输入特征
age = st.number_input("Age:", min_value=12, max_value=72, value=50, step=1)
symptom_duration = st.number_input("Symptom Duration(days):", min_value=1, max_value=360, value=25, step=1)
flexion = st.number_input("Flexion:", min_value=15, max_value=60, value=40, step=1)
extension = st.number_input("Extension:", min_value=15, max_value=50, value=36, step=1)
rotation_rom = st.number_input("Rotation:", min_value=35, max_value=80, value=50, step=1)
spring_test_pain = st.selectbox("Spring test-pain (0=No, 1=mild, 2=middle, 3=severe):", options=[0, 1, 2, 3])
muscle_tightness = st.selectbox("Muscle tightness (0=No, 1=mild, 2=middle, 3=severe):", options=[0, 1, 2, 3])
exacerbation_on_flexion = st.selectbox("Exacerbation on Flexion (0=No, 1=Yes):", options=[0, 1])
exacerbation_on_extension = st.selectbox("Exacerbation on Extension (0=No, 1=Yes):", options=[0, 1])

# 整合特征
feature_values = [
    age, symptom_duration, flexion, extension, rotation_rom,
    spring_test_pain, muscle_tightness, exacerbation_on_flexion, exacerbation_on_extension,
]
features = pd.DataFrame([feature_values], columns=feature_names)

# 预测按钮

if st.button("Prediction"):
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]

    # Generate advice based on prediction results
    probability = predicted_proba[predicted_class] * 100

    if predicted_class == 1:
        advice = (
            f"According to the model, your probability of benefiting from spinal manipulation is {probability:.1f}%."
        )
    else:
        advice = (
            f"According to the model, your probability of not benefiting from spinal manipulation is {probability:.1f}%."
        )

    # 调整字体大小
    font_size = "18px"  # 可根据需要设置字体大小
    advice_html = f"""
    <div style="font-size: {font_size};">
        {advice}
    </div>
    """
    st.markdown(advice_html, unsafe_allow_html=True)

    # SHAP 分析
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(features)

    # 获取 SHAP 值，并分别计算正负两边贡献值最大的特征
    shap_values_single = shap_values[0]  # 当前样本的 SHAP 值
    top_positive_idx = shap_values_single.argsort()[-3:][::-1]  # 贡献值最大的 3 个正向特征
    top_negative_idx = shap_values_single.argsort()[:3]  # 贡献值最大的 3 个负向特征

    # 构造特征名称列表，只保留正负两边前两名特征，其余为空字符串
    top_features = set(top_positive_idx).union(set(top_negative_idx))
    display_feature_names = [
        feature_names[i] if i in top_features else "" for i in range(len(feature_names))
    ]

    # 生成并显示静态 SHAP 力图
    shap.force_plot(
        base_value=explainer.expected_value,
        shap_values=shap_values_single,
        features=display_feature_names,  # 替换为正负两边前两名特征名称
        feature_names=None,  # 不显示具体特征值
        matplotlib=True
    )
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=300)
    st.image("shap_force_plot.png")


# # 计算期望值：获取类别 0 的期望值
# predictions = model.predict_proba(features)  # 获取模型对所有样本的预测概率
# expected_value = np.mean(predictions, axis=0)[0]  # 获取类别 0 的期望值
#
# # 提取类别 0 的 SHAP 值
# sample_shap_values_class_0 = shap_values[0][0, :]  # 提取类别 0 的 SHAP 值，第一个样本
# # 创建 SHAP 解释对象（SHAP 0.39.x 或更低版本）
# explanation = shap.Explanation(
#     values=sample_shap_values_class_0,  # 类别 0 的 SHAP 值
#     base_values=expected_value,  # 类别 0 的期望值
#     data=features.iloc[0].values,  # 第一个样本的特征值
#     feature_names=feature_names  # 特征名称
# )
#
# # Save the SHAP force plot as HTML
# shap.save_html("shap_force_plot.html", shap.plots.force(explanation, show=False))
#
# # Display the saved HTML in Streamlit
# st.subheader("模型预测的力图")
# with open("shap_force_plot.html", "r", encoding="utf-8") as f:
#     st.components.v1.html(f.read(), height=800)


# # 计算 SHAP 值
#     # 计算 SHAP 值
#     explainer = shap.KernelExplainer(model.predict_proba, features)  # 使用 model.predict_proba 作为解释器
#     shap_values = explainer.shap_values(features)
#     # 提取单个样本的 SHAP 值和期望值
#     sample_shap_values = shap_values[0]  # 提取第一个样本的 SHAP 值
#     expected_value = explainer.expected_value[0]  # 获取对应输出的期望值
#
#     #  shap_values = explainer.shap_values(features)
#     # shap.force_plot(
#     #     explainer.expected_value[0], shap_values[0][0], features.iloc[0], feature_names=feature_names
#     # )
# #
#     # 创建 Explanation 对象
#     explanation = shap.Explanation(
#         values=sample_shap_values[:, 0],  # SHAP 值
#         base_values=expected_value,  # 期望值
#         data=features.iloc[0].values,  # 输入数据
#         feature_names=feature_names  # 特征名称
#     )
#
#     explanation = shap.Explanation(values=sample_shap_values[:, 0], base_values=expected_value,
#                                      data=features.iloc[0].values, feature_names=feature_names)
#
# # 选择特定输出的 SHAP 值
#     # 保存为 HTML 文件
#     shap.save_html("shap_force_plot.html", shap.plots.force(explanation, show=False))
#     # 在 Streamlit 中显示 HTML
#     st.subheader("模型预测的力图")
#     # 读取 HTML 文件时指定编码
#     with open("shap_force_plot.html", "r", encoding="utf-8") as f:
#         st.components.v1.html(f.read(), height=600)
    # # Background data (using sample data for demonstration purposes)
    # # Ideally, use a representative background data (e.g., training data or random samples)
    # background_data = pd.DataFrame([feature_values] * 100, columns=feature_names)
    #
    # # KernelExplainer requires the model's predict_proba function and background dataset
    # explainer = shap.KernelExplainer(model.predict_proba, background_data)
    #
    # # Compute SHAP values for the given input features
    # shap_values = explainer.shap_values(features)
    #
    # # Check if SHAP values are correctly generated
    # if shap_values is None or len(shap_values) != 2:
    #     st.error("SHAP values are not correctly generated.")
    # else:
    #     # SHAP values generated successfully, display force plot
    #     shap.initjs()  # Initialize JS for SHAP visualization
    #     force_plot_html = shap.force_plot(explainer.expected_value[1], shap_values[1], feature_values,
    #                                       feature_names=feature_names)
    #     st.components.v1.html(force_plot_html, height=600)

    # # LIME explanation
    # explainer = lime.lime_tabular.LimeTabularExplainer(
    #     training_data=np.random.rand(100, len(feature_names)),  # Replace with actual training data
    #     training_labels=np.random.randint(0, 2, 100),  # Replace with actual training labels
    #     feature_names=feature_names,
    #     class_names=['No Benefit', 'Benefit'],
    #     mode='classification'
    # )
    #
    # # Predict and explain
    # predicted_class = model.predict(features)[0]
    # predicted_proba = model.predict_proba(features)[0]
    #
    # # Display prediction results
    # st.write(f"**Predicted Class:** {predicted_class}")
    # st.write(f"**Prediction Probabilities:** {predicted_proba}")
    #
    # # Generate explanation
    # exp = explainer.explain_instance(features[0], model.predict_proba)
    #
    # # Show LIME explanation as a table
    # st.write("**LIME Explanation of the Model Prediction:**")
    # exp.show_in_notebook()
    #
    # # Or visualize the explanation
    # fig = exp.as_pyplot_figure()
    # st.pyplot(fig)