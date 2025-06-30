# 🎓 Student Dropout Predictor | LightGBM + SHAP + Elite ML Workflow

> A high-performance, interpretable machine learning model to identify students at risk of dropping out — built with industry-grade techniques, optimized algorithms, and deep statistical insight.


## 🚀 Preview First — What This Project Delivers

### 🧠 Predict Student Dropout Risk

```python
example_student = {
    'age': 17, 'Medu': 4, 'Fedu': 3, 'traveltime': 1,
    'studytime': 2, 'failures': 0, 'famrel': 4, 'freetime': 3,
    'goout': 2, 'Dalc': 1, 'Walc': 2, 'health': 4, 'absences': 2,
    'school_MS': 1, 'sex_M': 1, 'address_U': 1, 'famsize_LE3': 0,
    'Pstatus_T': 1, 'Mjob_services': 1, 'Fjob_other': 1,
    'reason_reputation': 1, 'guardian_mother': 1,
    'famsup_yes': 1, 'activities_yes': 1, 'nursery_yes': 1,
    'higher_yes': 1, 'internet_yes': 1, 'romantic_yes': 0
}

predict_dropout(example_student)


## 🔍 Model Interpretability

https://raw.githubusercontent.com/raduan-alam/student-dropout-predictor/main/shap_summary_dropout_predictor.png
</p>


> Interpretable patterns that decision-makers and interviewers love – showing how each feature contributes to dropout risk.



## 🧪 Highlights: What Makes This Project Special

| Category         | What’s Used                                            |
|------------------|--------------------------------------------------------|
| **Model**        | LightGBM (boosted trees)                               |
| **Tuning**       | `GridSearchCV` for hyperparameter optimization         |
| **Prep**         | Null handling, encoding, scaling, class balance        |
| **Interpretation** | SHAP summary + force plots                            |
| **Deployment Ready** | `.pkl` model + `predict_dropout()` interface          |
| **Performance**  | 82.3% Accuracy | 86.7% ROC-AUC | Cross-validated        |

> This isn’t just a basic ML project — it demonstrates understanding, discipline, and professional readiness.


## 📁 Repository Structure

| File                       | Purpose                                                          |
|----------------------------|------------------------------------------------------------------|
| `student_dropout_predictor.ipynb` | Full notebook: loading → cleaning → modeling → interpretation |
| `dropout_model.pkl`        | Final trained LightGBM model, ready for deployment               |
| `shap_summary_plot.png`    | SHAP visualization for global feature importance                 |
| `requirements.txt`         | Clean, minimal dependency list                                   |
| `README.md`                | The file you're reading now 🔥                                   |


## 📉 Model Metrics

- ✅ Accuracy: `82.3%`
- ✅ ROC-AUC: `86.7%`
- 🔁 K-Fold Cross-Validation: Used to avoid overfitting
- ⚖️ Handles class imbalance and missing values gracefully


## 📚 Dataset Overview

- **Source**: [UCI Student Performance Dataset](https://archive.ics.uci.edu/ml/datasets/student+performance)
- **Features**: Academic performance, family background, lifestyle, support systems, and behavior
- **Target**: Dropout status (binary classification)


## 💡 Core Skills Demonstrated

- End-to-End Machine Learning Workflow  
- Advanced Tree-Based Modeling (LightGBM)  
- SHAP-Based Model Explainability  
- Feature Engineering & Encoding  
- Cross-Validation & Hyperparameter Tuning  
- Model Deployment Preparation (Pickle, Custom Predict Function)
