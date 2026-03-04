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
│   └── model_comparison_results.csv               # Model Results
├── requirements.txt
└── README.md
```

## Modeling General Approach

1. Baseline Model (No Class Balancing): Establish a reference point using default training behavior

2. Balanced Model (With Class Balancing): Improve sensitivity to the minority class (Purchase=1)

3. Hypertuned Model
- Find the best hyperparameter combination including balancing strategy
- Optimize via F2 score across 5-fold cross-validation to capture most potential consumers

4. Threshold-Tuned Model
- Tune the decision threshold for deployment
- Prioritize recall of buyers (Purchase=1) by optimizing its F2 score while preventing precision from dropping below 0.5

5. Retrain and Final Evaluation:
- Retrain threshold-tuned models using training and validation sets combined, more data = better generalization
- Evaluate model performance on test set to select the best model

## Model Performance

### Algorithm/Model Comparison Summary

**Logistic Regression**

| Stage | AUC | AUPRC | Recall | Precision | F2 | Buyers Caught |
|-------|-----|-------|--------|-----------|-----|---------------|
| Baseline | 0.9038 | 0.6346 | 0.56 | 0.66 | 0.5784 | 107/191 |
| Balanced | 0.9056 | 0.6074 | 0.79 | 0.51 | 0.7136 | 151/191 |
| Hypertuned | 0.9070 | 0.6234 | 0.76 | 0.53 | 0.6991 | 145/191 |
| Threshold-Tuned | 0.9070 | 0.6234 | 0.81 | 0.50 | 0.7229 | 155/191 |

**Random Forest**

| Stage | AUC | AUPRC | Recall | Precision | F2 | Buyers Caught |
|-------|-----|-------|--------|-----------|-----|---------------|
| Baseline | 0.9026 | 0.6770 | 0.57 | 0.66 | 0.5825 | 108/191 |
| Balanced | 0.9019 | 0.6647 | 0.54 | 0.66 | 0.5640 | 104/191 |
| Hypertuned | 0.9136 | 0.6986 | 0.73 | 0.54 | 0.6836 | 140/191 |
| Threshold-Tuned | 0.9136 | 0.6986 | 0.81 | 0.50 | 0.7203 | 154/191 |

**XGBoost**

| Stage | AUC | AUPRC | Recall | Precision | F2 | Buyers Caught |
|-------|-----|-------|--------|-----------|-----|---------------|
| Baseline | 0.9108 | 0.6790 | 0.54 | 0.65 | 0.5634 | 104/191 |
| Balanced | 0.9184 | 0.7058 | 0.80 | 0.51 | 0.7150 | 152/191 |
| Hypertuned | 0.9173 | 0.7052 | 0.80 | 0.51 | 0.7143 | 152/191 |
| Threshold-Tuned | 0.9173 | 0.7052 | 0.80 | 0.51 | 0.7170 | 152/191 |

**Final Models (Retrained on Train + Val, Evaluated on Test)**

| Algorithm | AUC | AUPRC | Recall | Precision | F2 | Buyers Caught |
|-----------|-----|-------|--------|-----------|-----|---------------|
| Logistic Regression | 0.9232 | 0.6636 | 0.83 | 0.52 | 0.7413 | 317/381 |
| Random Forest | 0.9326 | 0.7408 | 0.86 | 0.54 | 0.7662 | 327/381 |
| XGBoost | 0.9338 | 0.7356 | 0.84 | 0.54 | 0.7567 | 321/381 |

### Selected Model: Random Forest

**Model Specifications:**
- Class balancing: class_weight = 'balanced'
- Hyperparameters (best average precision): n_estimators = 200, max_depth = 10, max_features = 'sqrt', min_samples_leaf = 4, min_samples_split = 2
- Decision threshold (best F2): 0.394

**Performance Metrics:**
- AUC-ROC: 0.9297
- Recall: 0.86
- Precision: 0.53
- F2 Score: 0.7613 (emphasizes recall over precision)

**Selection Rationale:**
1. Catching the most buyers
- Lowest number of potential buyers undetected (false negatives)
- Best F2 score for buyers (0.7613), optimizing recall for catching most buyers

2. Very high AUC (0.9297), great at overall probability ranking ability

2. Best ROI even with relatively low precision (0.53)
- Assume that the false postives in this model are advertisements/discounts sent to non-buyers
- The revenue gained from the selected model is optimized with the highest recall which outweighs the trivial cost of advertising in the e-commerce world
- What we truly want to optimize is the recall which represents how many potential buyers are reached

**Confusion Matrix:**
```
                Predicted
              No Buy  |  Buy
Actual No Buy  1794   |  291  
Actual Buy      55    |  326  
```

**Top 3 Most Important Features:**
1. pagevalue_exit_interaction (0.229) - Interaction between page value and exit behavior
2. PageValues (0.216) - Historical average value of pages visited
3. has_pagevalue (0.149) - Binary indicator: visited any high-value pages

**Page Value is the single most dominant predictor**

## Limitations

1. **Low Precision (53%):** Nearly half of predicted buyers are false positives, meaning promotional costs could be high if intervention is expensive

2. **Dataset Age:** Data is from 2018-2019, consumer behavior patterns may have shifted post-pandemic

3. **Lack of Product Context:** Model doesn't account for product categories, prices, or inventory availability that could affect purchase decisions

4. **No User History:** Each session is treated independently, and incorporating user purchase history could improve predictions for returning visitors

5. **Geographic Limitations:** Dataset is from a single region, so model may not generalize to markets with different shopping behaviors

## Tech Stack

- **Python 3.13.11**
- **Data Processing:** pandas, numpy
- **Machine Learning:** scikit-learn, XGBoost
- **Visualization:** matplotlib, seaborn, SHAP
- **Development:** Jupyter notebooks, VS Code

## Dataset Source
```
Sakar, C.O., Polat, S.O., Katircioglu, M. and Kastro, Y., 2019. 
Real-time prediction of online shoppers' purchasing intention using multilayer 
perceptron and LSTM recurrent neural networks. 
Neural Computing and Applications, 31(6), pp.6893-6908.
```