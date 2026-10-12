# Demand Forecasting Using Machine Learning

## Project Overview
This project forecasts warehouse product demand using historical transactional data from Kaggle.

## Objectives
- Clean and preprocess transactional data.
- Aggregate demand by day.
- Engineer temporal features, including lag and rolling-average features.
- Detect and remove outliers using Z-score filtering.
- Train and evaluate Random Forest and PyTorch MLP regression models.
- Compare prediction accuracy and asymmetric business penalties.

## Methodology
1. Data ingestion and cleaning
2. Daily demand aggregation
3. Feature engineering
4. Outlier detection
5. Chronological train-test split
6. Standard scaling
7. K-Means demand-regime clustering
8. Random Forest and PyTorch MLP training
9. Model evaluation

## Evaluation Metrics
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Asymmetric business penalty, with a 2x penalty for under-prediction

## Technologies Used
Python, Pandas, NumPy, Scikit-learn, PyTorch, Matplotlib, Git, and GitHub.

## Results
Add your actual test MAE, RMSE, asymmetric penalty values, and model comparison after running the experiments.

## Project Structure
Describe the notebooks, source code, results, and flowchart included in this repository.

## Dataset
The project uses the Kaggle transactional demand dataset. Follow the dataset's licensing and access requirements; do not redistribute the raw data unless permitted.
