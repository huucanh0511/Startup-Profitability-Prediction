<div align="justify">
  
# About this Repository ℹ️

This repository contains the project I did as a part of the coursework for ECS784P - Data Analytics. <br>
The assignment required me to address a data-related problem in a chosen field of interest by applying at least two data analytics techniques from a given list of machine learning algorithms.


- Linear, non-linear and logistic regression
- Support vector classification or regression
- Decision trees, with or without random forest
- KNN
- k-means
- GMMs

<br>

## Motivation ⚙️
### Chosen data-related problem
<br>

> "How accurately can machine learning models predict startup profitability based on funding, market share, and other business factors?"

<br>

Given the high failure rates of startups and the increasing adoption of ML in financial prediction, this study focuses on predicting startup profitability—a key financial outcome that directly influences long-term survival. Unlike previous studies that broadly assess startup success, this research aims to provide a more precise financial evaluation by examining the ability of startups to generate profit based on factors such as market share, funding, revenue, and operational scale. By identifying the most influential predictors of profitability, this study seeks to enhance data-driven decision-making for investors, entrepreneurs, and market analysts.

<br>

### Chosen dataset 

The dataset used was [Startup Growth & Funding Trends](https://www.kaggle.com/datasets/samayashar/startup-growth-and-funding-trends), which included features such as funding rounds, funding amount, valuation, revenue, employees, market share, industry, and exit status. The target variable was profitability (binary: 0 = not profitable, 1 = profitable).

<br>

### Chosen machine learning algorithms

1. Logistic Regression

    - Interpretable and Efficient

      - Logistic Regression is a simple, linear classification model that provides clear insights into feature importance.

    - Baseline Model
  
      - It serves as a strong benchmark model to compare against more complex algorithms.

    - Works well for binary classification

      - Since profitability is a binary outcome (Profitable = 1, Not Profitable = 0), Logistic Regression is well-suited for this task.

    - Handles small Datasets well

      - Given the dataset size (500 startups), Logistic Regression performs efficiently without the risk of overfitting.

<br>

- Compared to k-means and GMM (unsupervised models), Logistic Regression is a better choice.

  - These are clustering algorithms and are not suitable for direct classification tasks.

- Compared to Decision Trees (single trees), Logistic Regression is a better choice.

  - Logistic Regression is less prone to overfitting, making it a better baseline model.

<br>

2. Random Forest

  - Captures non-linear relationship

    - Unlike Logistic Regression, Random Forest can model complex interactions between financial factors affecting profitability.

  - Less prone to overfitting than single decision trees

    - Random Forest reduces overfitting by averaging multiple trees, making it more robust than individual decision trees.

  - Handles mixed data well

    - Random Forest works well with a mix of continuous and categorical variables, which suits the dataset's financial and market-related features.

  - Feature importance ranking

    - It provides insights into which financial indicators are most influenced, helping refine feature selection.

<br>

- Compared to Support Vector Machines (SVM), Random Forest is a better choice.

  - SVM requires careful parameter tuning and scales poorly with larger feature sets, making Random Forest a more practical choice.

- Compared to k-Nearest Neighbours (KNN), Random Forest is a better choice.

  - KNN struggles with high-dimensional data and large datasets, whereas Random Forest performs well in such cases.

- Compared to linear and non-linear regression, Random Forest is a better choice.

  - Regression models predict continuous values, whereas Random Forest is better suited for classification.

<br>

The models were trained using an 80:20 train-test split, and performance was evaluated using accuracy, AUC score, precision, recall, and F1-score.

<br>

### Project Objectives 🎯

The project aimed to evaluate whether machine learning models can effectively classify startups as profitable or unprofitable using key financial and operational metrics.

<br>

1. Identify the key variables influencing startup profitability predictions.
2. Develop and assess two machine learning models to classify startups as profitable or
unprofitable.
3. Conduct Exploratory Data Analysis (EDA) to uncover patterns, correlations, and distributions.
4. Uilise Python libraries such as Pandas, Seaborn, and Matplotlib to visualise trends and present findings effectively.
5. Evaluate model performance using AUC as the primary metric, supplemented by accuracy, recall, and F1-score for a comprehensive analysis.
6. Analyse the practical implications for investors and entrepreneurs, proposing refinements to enhance predictive accuracy.

<br>

## Environment 👩🏻‍💻 

<div align="center">
  <img src="https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=flat-square&logo=jupyter&logoColor=white">&nbsp;&nbsp;&nbsp;&nbsp;
    
<img src="http://img.shields.io/badge/Visual%20Studio%20Code-eeeeee.svg?style=flat-square&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAB4AAAAeCAYAAAA7MK6iAAAFU0lEQVRIx+1Wa2iWZRi+nuc9fedt7uCnrth0y9QMzVTSLFxhs8iaEFkRCwrKisSCIEgUifpjICFRRgc6iKClleWkxEjRskxHbvPUnC530O3b9p2/932e++7HNw+luP0YCNUDN++fh/t67+u+rvu5BTPjWhyJa3T+e8Dinz3+srkPmw/3w2cZWDqrGD+cTGJhdQggQnNXHOMKAygJ+xBLprFhfzvKwj6MK/RjVNAZs+tE7I2kVTh6fGlkXdBQPyY9I6GYMXOMidrxgZGtWEoBITDpw32nvtx+8GR9S3e69qceY1tLv9zqMeoNgaAQI0y1zGecvuVgx+YjHQMzQ44JWwoAAn8kRM1vveKjxpjYSYT5IwLMAPyWAUOKudsaOz4/1h2fHLAlNAOaGQzANgQUAX2umP3JUb12WMBCAFIAhrycIwEgaJvY2xqr3XP83MaeRLbSMSU0MZQiENHf8tgGkFIiMiSwzxRQGg/Ec7QumdPltiHAlyQKORZ2NHXWr/3++Ka+lFvusw1oYmgipF0FhlCaAU2AYsDL/wcNCdx0Nvf4oa7sx7tPJZ9bsbPr24Ec3Rtx8tf8loEvDv65bE1D83pmDjmmAWZAaUJWERZOjb4/p6p4fcrV0ACI86GvMB0vA/6xLflYT0ZFIo7EiVhu6ieNfRs3NcVf8FvS2n20+5X1u0+s8YhtQwCKCPGsBwLceVUlr94/dczS0pC/NacYRAARoBkgghjSx3taYxOXN5x9/0xCzQ1aAq5mkBD6xgJ5pO30qUmkPGlZJogY8azC+JLQsedrql4805f5pl+E4DkFq/Z08UrfYIuIAb/JrQ114QlXrVgIHL056jwyu9y3M+kSLClgAkZLH01R4aiUdgCep9Gf8TCvumTb2iXT7qsqDXyjLT9koBCKiDUBRHwxeBg+Lg+bmBW12m8v9y26oyK4Ja0YQgCOAQjTgRceDeVE8MydE1Y8NOO6uori4ImEC2TMMAT0BbvpS0JdgerLgG1T4voCG0dOd4dine1hlRqAZgFFDKU1NAl4/mKc5YKSHu2EcopxOsFIugI6r13WfFFYg6GHAWygYyAzZV9rz44Dbb1325leuIkYki7BUwwwQYDx2aHYsk2HB75r7MrMNSUQEhmMDUmkPTJo0E6aBoHBQ1fc2D4w980dLZ+f6klN89kmkjkPMt2bnl6CXYYh0ymPQEwI2QI/t6dufX7bma+bzuWeqh5lYUKhCUX5iZWfYnkvD4vq17cfWd3SGZ8Y9pnoS7soKwh03ndzeX1Nhb9m+ZzSp13FiaRL0AyEbIlYRhW9tffcez+0pd6JOl5RNCjj7nkbYfDLwwCeWVm8pros1HkunsW86rJdL9dOXhTymZtTLmF+ZfDT1XeVPRm2jVgiq6GJYRoCjinxwYHep1/a3rGhrd+daUoBzQw9qGoweEgfr/z6MCQwK+fx9Po5FRuauhKJ75u7URQOYOFNY1FZZOGXM5napV+1f9idVNECR4IBsACSWYVx0TIUjSqGpryeiIGAiZO/PhoZf9WKlWY4prE/WuB7128biZwinH9PGUA8q/Hg5EjDstuKF08YZR9LuJQvh4GgJQFmaOYLiub8vDaH9ToRMxTlqbrSk+jmvb3vsWlFddUlzsF4ji5yeUk3NQMpj1HsQ+eILQKaAE9z87OzSxcvqArtSmQUmAHBDE8zBrIER3J6UhG2vjLDempENxBNDGa0raoZ/ciCGyINyZxCymM4Eql7rsPmukqxeP44WSeA30d8y9TMUITu1XePXfLEraNfu2Ws7+1FFbzggQrxsG1ghzs4RIZU9f8L/b8O+C9rntGn+AUx4QAAAABJRU5ErkJggg==">
</div>

<br>

## Stack 🛠️
<div align="center">
  <img src="https://img.shields.io/badge/python-3670A0?style=flat-square&logo=python&logoColor=white">
</div>

<br>

## Repository Structure 🌲
```bash
.
├── .gitattributes
├── Startup_Profitability_Prediction_Report_v3.pdf
├── Startup_Profitability_Prediction_v3.ipynb
├── startup_data.csv
└── README.md
```

<br>

## Results 

<br>

Logistic Regression (Before vs. After hyperparameter tuning)

| Metric                 | Before Tuning  | After Tuning  | Change  |
|------------------------|---------------|--------------|------------|
| **Training Accuracy**  | 0.5925        | 0.5925       | No change  |
| **Test Accuracy**      | 0.5400        | 0.5400       | No change  |
| **Training AUC**       | 0.5927        | 0.5927       | No change  |
| **Test AUC**           | 0.5051        | 0.5051       | No change  |
| **Mean CV Accuracy**   | 0.5600        | 0.5600       | No change  |
| **Mean CV AUC**        | 0.5490        | 0.5490       | No change  |
| **False Negative Rate**| 74.4%         | 74.4%        | No change  |

<br>

- Hyperparameter tuning did not improve performance for Logistic Regression.

- The model remains underfitted, failing to capture useful patterns.

- High false negative rate (74.4%) – meaning many profitable startups were misclassified as unprofitable.

<br>

Random Forest (Before vs. After hyperparameter tuning)

| Metric                 | Before Tuning  | After Tuning  | Change  |
|------------------------|---------------|--------------|---------|
| **Training Accuracy**  | 1.0000        | 0.5775       | Reduced (Overfitting mitigated) |
| **Test Accuracy**      | 0.5200        | 0.5700       | Improved |
| **Training AUC**       | 1.0000        | 0.6183       | Reduced (Overfitting mitigated) |
| **Test AUC**           | 0.5204        | 0.5333       | Slightly improved |
| **Mean CV Accuracy**   | 0.5875        | 0.5425       | Slight decline |
| **Mean CV AUC**        | 0.5789        | 0.5589       | Slight decline |
| **False Negative Rate**| 62.8%         | 53.5%        | Improved (Fewer profitable startups misclassified) |


<br>

- Hyperparameter tuning reduced overfitting in Random Forest.
- Test accuracy improved from 52% → 57%, meaning better generalisation.
- False negative rate improved, making the model better at correctly identifying profitable startups.
- However, AUC score remained low (≈ 0.53), indicating limited predictive power.

<br>

## Main Findings 🔍

<br>

1. Market Share (%) emerged as the strongest predictor of startup profitability in both models.

    - Indicates that a higher market share significantly increases profitability likelihood.
  
    - However, its dominance suggests potential dataset bias toward a single feature.

2. Funding Rounds was a strong predictor in Logistic Regression but weak in Random Forest.

    - Implies a linear impact with limited interactions.
      
3.  The number of Employees was more influential in Random Forest

    - Suggests a non-linear relationship where specific workforce sizes may enhance profitability

4. Revenue and Funding Amount proved to be weak predictors.

    - Indicates that capital alone does not determine profitability.

<br>

## Recommendations for Improvements 📈

<br>

1. Feature Engineering Enhancements
  
    - Incorporate historical financial trends, cost structures, and customer-related metrics for improved predictions and more holistic assessment of startup profitability
    - Integrate external economic and industry trend data
    
2. Advanced Modelling Approaches
  
    - Explore Gradient Boosting (XGBoost, LightGBM) to capture non-linear relationships
    - Reformulate the problem as a regression task to predict profit margins instead of binary profitability
      
3. Data Balancing & Augmentation
   
   - Address class imbalance using oversampling or synthetic data techniques (SMOTE)

<br>

## Reflection 🪞

<br>

This project provided valuable insights into the challenges of predicting startup profitability using machine learning. Through the implementation of Logistic Regression and Random Forest, I gained a deeper understanding of model performance evaluation, feature importance analysis, and the impact of data quality on predictive accuracy.

I improved my skills in data preprocessing, feature selection, and hyperparameter tuning, particularly in understanding how different models interpret financial data. Additionally, working with AUC, precision, recall, and F1-score deepened my knowledge of model evaluation beyond simple accuracy metrics.

Moving forward, I aim to further develop my expertise in advanced machine learning techniques, particularly ensemble methods like XGBoost and LightGBM, which may better capture non-linear relationships in financial prediction. Additionally, I want to explore Explainable AI (e.g., SHAP, LIME) to enhance model interpretability, ensuring that machine learning can provide more actionable insights for investors and entrepreneurs.

This project highlighted the complexity of predicting business success, and I am keen to further study financial data science, feature engineering techniques, and real-world applications of predictive modelling to enhance my analytical capabilities.

<br>
  
</div>




