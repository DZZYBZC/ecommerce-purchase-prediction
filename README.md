# E-commerce Purchase Prediction

## Project Overview

This project develops a machine learning system to predict online shoppers' purchasing intention. Using behavioral session data from 12,330 unique user sessions, the model identifies which visitors are likely to complete a purchase, enabling targeted conversion optimization strategies.

## Project Goal

Build a classification model that:
1. Maximizes recall (catches as many potential buyers as possible)
2. Maintains acceptable precision (at least 50%) to avoid excessive false positives
3. Provides interpretable predictions to understand purchase drivers

**Dataset:** Online Shoppers Purchasing Intention Dataset from UCI Machine Learning Repository (Sakar et al., 2019)

## Repository Structure
```
ecommerce-purchase-prediction/
├── data/
│   └── online_shoppers_intention.csv              # Raw dataset (12,330 sessions)
├── notebooks/
│   ├── 01_eda.ipynb                               # Exploratory data analysis
│   ├── 02_feature_engineering_and_modeling.ipynb  # Model development
│   └── 03_interpretability.ipynb                  # SHAP analysis
├── reports/
│   ├── figures/                                   # SHAP visualizations
│   │   ├── shap_feature_importance_bar.png
│   │   ├── shap_summary_beeswarm.png
│   │   ├── shap_waterfall_confident_buyer.png
│   │   └── shap_waterfall_false_negative.png
│   └── model_comparison_results.csv               # Performance metrics
├── requirements.txt
└── README.md
```

## Modeling General Approach

1. Baseline Model (No Class Balancing): Establish a reference point using default training behavior.

2. Balanced Model (With Class Balancing): Improve sensitivity to the minority class (Purchase=1).

3. Hyperparameter-Tuned Model: Find the best hyperparameter combination for classification performance under class imbalance based on Average Precision which balances precision and recall and works well for our imbalanced dataset.

4. Threshold-Tuned Model: Tune the decision threshold for deployment, prioritizing recall of buyers (Purchase=1) by optimizing its F2 score while preventing precision from dropping below 0.5.

## Model Performance

### Algorithm/Model Comparison Summary

| Algorithm | Stage | AUC | Recall | Precision | F2 | Buyers Caught |
|-----------|-------|-----|--------|-----------|-----|---------------|
| Logistic Regression | Baseline | 0.9135 | 0.57 | 0.70 | 0.5889 | 324/572 |
| Logistic Regression | Balanced | 0.9160 | 0.80 | 0.53 | 0.7272 | 460/572 |
| Logistic Regression | Hypertuned | 0.9076 | 0.78 | 0.55 | 0.7210 | 448/572 |
| Logistic Regression | Threshold | 0.9076 | 0.81 | 0.52 | 0.7255 | 461/572 |
| Random Forest | Baseline | 0.9140 | 0.60 | 0.70 | 0.6225 | 346/572 |
| Random Forest | Balanced | 0.9149 | 0.57 | 0.71 | 0.5948 | 327/572 |
| Random Forest | Hypertuned | 0.9247 | 0.73 | 0.60 | 0.7019 | 420/572 |
| Random Forest | Threshold | 0.9247 | 0.87 | 0.51 | 0.7596 | 496/572 |
| XGBoost | Baseline | 0.9226 | 0.58 | 0.70 | 0.6010 | 332/572 |
| XGBoost | Balanced | 0.9285 | 0.82 | 0.53 | 0.7406 | 471/572 |
| XGBoost | Hypertuned | 0.9288 | 0.82 | 0.52 | 0.7381 | 470/572 |
| XGBoost | Threshold | 0.9288 | 0.86 | 0.50 | 0.7560 | 494/572 |

### Selected Model: Random Forest (Threshold-Tuned)

**Model Specifications:**
- Algorithm: Random Forest with 200 trees (n_estimators = 200)
- Class balancing: class_weight = 'balanced'
- Other hyperparameters: max_depth = 30, max_features = 'sqrt', min_samples_leaf = 4, min_samples_split = 10
- Decision threshold: 0.291 (tuned for F2 optimization)

**Performance Metrics:**
- AUC-ROC: 0.9247
- Recall: 0.87 (catches 496 out of 572 actual buyers = 86.7%)
- Precision: 0.51
- F2 Score: 0.7596 (emphasizes recall over precision)

**Selection Rationale:**
1. Highest recall among all models (0.87)
2. Best F2 score (0.7596), optimizing for recall while maintaining precision >= 0.5
3. Superior business ROI: In e-commerce, the revenue from catching an additional buyer far outweighs the cost of showing a promotion to a non-buyer (false positive)
4. Catches 496/572 buyers compared to XGBoost's 494/572, despite similar AUC

**Confusion Matrix:**
```
                Predicted
              No Buy  |  Buy
Actual No Buy  2646   |  481  
Actual Buy      76    |  496  (87% recall)
```

**Top 5 Most Important Features:**
1. pagevalue_exit_interaction (0.229) - Interaction between page value and exit behavior
2. PageValues (0.216) - Historical average value of pages visited
3. has_pagevalue (0.149) - Binary indicator: visited any high-value pages
4. Month_Nov (0.034) - November visits (Black Friday effect)
5. ExitRates (0.070) - Average historical exit rate of visited pages

## Limitations

1. **Low Precision (51%):** Nearly half of predicted buyers are false positives, meaning promotional costs could be high if intervention is expensive

2. **Dataset Age:** Data is from 2018-2019, consumer behavior patterns may have shifted post-pandemic

3. **Lack of Product Context:** Model doesn't account for product categories, prices, or inventory availability that could affect purchase decisions

4. **No User History:** Each session is treated independently; incorporating user purchase history could improve predictions for returning visitors

5. **Geographic Limitations:** Dataset is from a single region; model may not generalize to markets with different shopping behaviors

## Technologies Used

- **Python 3.13.11**
- **Data Processing:** pandas, numpy
- **Machine Learning:** scikit-learn, XGBoost
- **Visualization:** matplotlib, seaborn
- **Interpretability:** SHAP
- **Development:** Jupyter notebooks, VS Code

## Dataset source
```
Sakar, C.O., Polat, S.O., Katircioglu, M. and Kastro, Y., 2019. 
Real-time prediction of online shoppers' purchasing intention using multilayer 
perceptron and LSTM recurrent neural networks. 
Neural Computing and Applications, 31(6), pp.6893-6908.
```