### **Heart Disease Prediction Using Machine Learning**

This project develops a machine learning model to predict the likelihood of a person having heart disease. The dataset, sourced from [Kaggle](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset), dates back to 1988 and combines data from four sources: Cleveland, Hungary, Switzerland, and Long Beach V. While the dataset contains 76 attributes, most published experiments focus on a subset of 14 key features. The "target" field indicates the presence of heart disease, where:
- **0**: No disease
- **1**: Disease present

---

### **Approach**

1. **Data Preparation**:
   - Split the dataset: 60% for training and 40% for testing.
   - Evaluated multiple machine learning models, both scale-sensitive and scale-insensitive, to determine the best fit.

2. **Model Selection**:
   - Chose the **RandomForestClassifier** for its superior performance:
     - Achieved a **recall score of 1.0**, ensuring all true positive cases were correctly identified.
   - Used **ROC Curve** and **ROC AUC Score** for additional evaluation:
     - The Random Forest model consistently outperformed others by comparing true positive rates and false positive rates.

3. **Hyperparameter Tuning**:
   - Performed hyperparameter optimization using **3-fold cross-validation** to refine the model. The optimal parameters for the Random Forest were:
     - **max_depth**: 10
     - **min_samples_leaf**: 2
     - **n_estimators**: 200
     - **n_jobs**: -1 (for parallel processing)

4. **Feature Importance Analysis**:
   - Extracted feature importances using the trained Random Forest model.
   - Visualized the results using **Matplotlib**, highlighting the most significant features: `cp`, `thal`, `oldpeak`, `thalach`, and `ca`.
   - Complemented this analysis with a **Seaborn heatmap** to explore feature correlations, providing deeper insights into the dataset.

---

### **Key Takeaways**
- The Random Forest Classifier was the best-performing model for predicting heart disease in this dataset.
- Features such as chest pain type (`cp`), `thal`, and `oldpeak` were critical predictors.
- The combination of feature importance and correlation analysis enabled a comprehensive understanding of the dataset.

---

### **Dataset**
You can access the dataset [here](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset).

---

### **Future Enhancements**
- Incorporate additional classifiers like XGBoost or LightGBM for comparison.
- Experiment with automated feature selection techniques to refine the input features.
- Enhance visualizations using interactive tools such as Plotly for better interpretability.

---

### **How to Use**
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/heart-disease-prediction.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the notebook or script:
   ```bash
   python heart_disease_prediction.py
   ```

   
