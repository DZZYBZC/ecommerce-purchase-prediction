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

1. Baseline Model (No Class Balancing): Establish a reference point using default training behavior.

2. Balanced Model (With Class Balancing): Improve sensitivity to the minority class (Purchase=1).

3. Hyperparameter-Tuned Model
- Find the best hyperparameter combination for classification performance using cross validation
- Base CV on Average Precision which balances precision and recall and works well for our imbalanced dataset

4. Threshold-Tuned Model
- Tune the decision threshold for deployment
- Prioritize recall of buyers (Purchase=1) by optimizing its F2 score while preventing precision from dropping below 0.5

5. Retrain and Final Evaluation:
- Retrain threshold-tuned models using training and validation sets combined
- Evaluate model performance on test set to select the best model

## Model Performance

### Algorithm/Model Comparison Summary

**Logistic Regression**

| Stage | AUC | Recall | Precision | F2 | Buyers Caught |
|-------|-----|--------|-----------|-----|---------------|
| Baseline | 0.9038 | 0.56 | 0.66 | 0.5784 | 107/572 |
| Balanced | 0.9056 | 0.79 | 0.51 | 0.7136 | 151/572 |
| Hypertuned | 0.8884 | 0.76 | 0.53 | 0.6991 | 145/572 |

**Random Forest**

| Stage | AUC | Recall | Precision | F2 | Buyers Caught |
|-------|-----|--------|-----------|-----|---------------|
| Baseline | 0.9026 | 0.57 | 0.66 | 0.5825 | 108/572 |
| Balanced | 0.9019 | 0.54 | 0.66 | 0.5640 | 104/572 |
| Hypertuned | 0.9134 | 0.71 | 0.57 | 0.6786 | 136/572 |

**XGBoost**

| Stage | AUC | Recall | Precision | F2 | Buyers Caught |
|-------|-----|--------|-----------|-----|---------------|
| Baseline | 0.9108 | 0.54 | 0.65 | 0.5634 | 104/572 |
| Balanced | 0.9184 | 0.80 | 0.51 | 0.7150 | 152/572 |
| Hypertuned | 0.9188 | 0.80 | 0.51 | 0.7176 | 153/572 |

**Final Models (Retrained on Train + Val, Evaluated on Test)**

| Algorithm | AUC | Recall | Precision | F2 | Buyers Caught |
|-----------|-----|--------|-----------|-----|---------------|
| Logistic Regression | 0.9172 | 0.77 | 0.57 | 0.7230 | 295/572 |
| Random Forest | 0.9297 | 0.86 | 0.53 | 0.7613 | 326/572 |
| XGBoost | 0.9339 | 0.84 | 0.54 | 0.7564 | 321/572 |

### Selected Model: Random Forest

**Model Specifications:**
- Class balancing: class_weight = 'balanced'
- Hyperparameters (best average precision): n_estimators = 200, max_depth = 30, max_features = 'sqrt', min_samples_leaf = 4, min_samples_split = 10
- Decision threshold (best F2): 0.291

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
- **Visualization:** matplotlib, seaborn
- **Interpretability:** SHAP
- **Development:** Jupyter notebooks, VS Code

## Dataset Source
```
Sakar, C.O., Polat, S.O., Katircioglu, M. and Kastro, Y., 2019. 
Real-time prediction of online shoppers' purchasing intention using multilayer 
perceptron and LSTM recurrent neural networks. 
Neural Computing and Applications, 31(6), pp.6893-6908.
```