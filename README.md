# Titanic Survival Predictor

A machine learning web application that predicts the survival probability of passengers aboard the RMS Titanic using a trained Random Forest model. The application provides an interactive interface built with Streamlit, allowing users to input passenger information and receive real-time survival predictions along with model insights.

## Overview

This project demonstrates a complete machine learning workflow, from data exploration and feature engineering to model training, evaluation, and deployment.

The model is trained on the Titanic dataset and deployed as an interactive web application. User inputs are processed through the same preprocessing and feature engineering pipeline used during training, ensuring consistency between training and inference.

## Features

* Interactive web interface built with Streamlit
* Real-time survival prediction
* Survival probability estimation using `predict_proba()`
* Automated feature engineering pipeline
* Feature importance visualization
* End-to-end machine learning deployment workflow

## Machine Learning Pipeline

### Feature Engineering

Several domain-specific features were created to improve predictive performance:

#### Title Extraction

Passenger titles are extracted from names and grouped into:

* Mr
* Mrs
* Miss
* Master
* Rare

#### Family Features

* `FamilySize = SibSp + Parch + 1`
* `IsAlone`
* `FamilyCategory` (Alone, Small, Large)

#### Cabin Features

* Deck extraction from cabin information
* Rare deck grouping

#### Ticket Features

* Ticket prefix extraction
* Rare ticket prefix grouping

#### Binned Features

Continuous variables are converted into categorical groups:

* AgeGroup
* FareBin

### Data Preprocessing

#### Numerical Features

* Missing value imputation using median values
* Standardization using `StandardScaler`

#### Categorical Features

* Missing value imputation using the most frequent category
* One-hot encoding with unknown-category handling

The preprocessing pipeline is implemented using `ColumnTransformer` and `Pipeline`.

### Model Selection

The following models were evaluated using stratified 5-fold cross-validation:

* Logistic Regression
* Random Forest Classifier
* Gradient Boosting Classifier
* Support Vector Machine (SVM)

### Hyperparameter Tuning

The best-performing Random Forest model was optimized using GridSearchCV.

Parameters tuned:

* Number of estimators
* Maximum tree depth

### Evaluation Metrics

The model was evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC Score
* Confusion Matrix

## Technologies Used

### Programming Language

* Python

### Machine Learning

* Scikit-learn
* NumPy
* Pandas

### Visualization

* Matplotlib
* Seaborn
* Plotly

### Web Application

* Streamlit

## Project Structure

```text
titanic-survival-predictor/
│
├── app.py
├── model.ipynb
├── titanic_model.pkl
├── requirements.txt
├── .gitignore
├── README.md
│
├── titanic/
│   ├── train.csv
│   ├── test.csv
│   └── gender_submission.csv
│
└── assets/
```

## Installation

### Clone the Repository

```bash
git clone https://github.com/deepakbandla/titanic-survival-predictor.git
cd titanic-survival-predictor
```

### Create a Virtual Environment

```bash
python -m venv .venv
```

Activate the environment:

#### Windows

```bash
.venv\Scripts\activate
```

#### Linux / macOS

```bash
source .venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Running the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your browser at:

```text
http://localhost:8501
```

## Model Export

The trained model pipeline is exported using Joblib:

```python
import joblib

joblib.dump(best_rf, "titanic_model.pkl")
```

Load the model for inference:

```python
import joblib

model = joblib.load("titanic_model.pkl")
prediction = model.predict(data)
```

## Results

The Random Forest classifier achieved strong predictive performance after feature engineering and hyperparameter tuning. Feature importance analysis showed that passenger class, fare, age, sex, and title were among the most influential factors affecting survival predictions.

## Future Improvements

* Add SHAP-based model explainability
* Support multiple model comparisons within the web app
* Deploy using Docker
* Add CI/CD workflow with GitHub Actions
* Track experiments using MLflow
* Add automated testing for inference pipelines

## Author

Deepak

GitHub: https://github.com/deepakbandla

LinkedIn: https://www.linkedin.com/in/deepak-bandla/
