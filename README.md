# 🎓 Student Dropout Predictor (LightGBM + SHAP)

> Predicting student dropouts with precision, transparency, and statistical rigor.

This project builds a powerful machine learning model to identify students at risk of dropping out using demographic, academic, and behavioral features. Designed with deployment in mind, it emphasizes performance, interpretability, and real-world utility — everything top universities and companies expect in applied ML.



## 🚀 Project Highlights

- ✅ **End-to-End ML Pipeline**: From raw CSV to tuned, interpretable prediction model  
- ⚡️ **Optimized LightGBM**: Tuned hyperparameters using `GridSearchCV` for elite performance  
- 📊 **SHAP Interpretation**: Model decisions explained clearly and visually  
- 🧠 **Statistical Rigor**: Null handling, feature selection, scaling, and cross-validation  
- 🧪 **Custom Prediction Interface**: Simple function to predict dropout risk for any student  



## 📁 Files Included

| File | Description |

| `student_dropout_predictor.ipynb` | Complete Jupyter notebook with code, preprocessing, modeling, interpretation |
| `dropout_model.pkl` | Final trained and optimized LightGBM model |
| `shap_summary_plot.png` | SHAP summary plot showing top predictive features |
| `requirements.txt` | Curated list of required Python packages (minimal and clean) |
| `README.md` | You’re reading it! |



## 📉 Performance

- **Model**: LightGBM (with hyperparameter tuning)  
- **Accuracy**: ~82.3% on test set  
- **ROC-AUC**: ~86.7%  
- **Cross-validated** for robustness  



## 🔍 SHAP Interpretability

Understanding *why* a student is predicted to drop out is as important as the prediction itself.

<p align="center">
  <img src="shap_summary_plot.png" width="700" alt="SHAP Summary Plot">
</p>

✅ This plot reveals the most influential features driving predictions — essential for educators and policy makers.



## 📦 Quick Usage

You can use the trained model directly with the `predict_dropout()` function:

```python
example_student = {
    'age': 17,
    'Medu': 4,
    'Fedu': 3,
    'traveltime': 1,
    'studytime': 2,
    'failures': 0,
    ...
    'romantic_yes': 0
}

predict_dropout(example_student)
Returns 1 if likely to drop out, 0 otherwise.



🧠 What You'll Learn From This Project
This repository demonstrates elite practices in applied machine learning:

Feature engineering and encoding for categorical data

Data standardization and preparation

Model benchmarking and optimization

SHAP visualizations for explainability

Exporting and reusing ML models in production

📚 Dataset
Source: UCI Student Performance Dataset

Data includes demographics, grades, study time, family background, and more

🧑‍💻 Author
Raduan Alam
Data Science & Machine Learning Enthusiast
🔗 GitHub Profile
