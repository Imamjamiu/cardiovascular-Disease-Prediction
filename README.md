# Prediction of Cardiovascular Disease Using Random Forest and KNN

This repository contains the implementation and supporting materials for a cardiovascular disease prediction research project that explored two supervised machine-learning algorithms: **Random Forest** and **K-Nearest Neighbors (KNN)**.

The project investigates how routinely collected health and lifestyle attributes can be used to build machine-learning models for cardiovascular disease classification. It is presented for research and educational purposes and is not a clinical diagnostic system.

## Project Contents

- `project (2).ipynb` — Jupyter Notebook containing the research workflow and experiments.
- `predictive system.py.py` — prediction-system implementation.
- `cvd prediction web app.py` — Streamlit web application.
- `cardiovascular_trained_model.sav` — saved trained model artifact.

## Methods

The original research compares Random Forest and KNN for cardiovascular disease prediction. Refer to the notebook and accompanying research material for the exact experimental procedure, preprocessing decisions, and reported results.

## Running the Application

This repository contains the original project files. To run the Streamlit application, first create a Python environment and install the dependencies required by the scripts. Then run:

```bash
streamlit run "cvd prediction web app.py"
```

The required package versions may depend on the environment in which the original model was trained.

## Related Engineering Implementation

A separate engineering project explores cardiovascular disease prediction with a modular Scikit-learn pipeline, Logistic Regression, and Streamlit.

[Explore CardioAI — the separate engineering implementation](https://github.com/imamjamiu/cardioai-cardiovascular-disease-prediction)

**Important:** CardioAI is a distinct implementation with a different workflow and model. Its methods and evaluation results should not be interpreted as results from this original Random Forest and KNN research project.

## Disclaimer

This repository is intended for research and educational purposes only. It is not a medical device and must not be used to diagnose, treat, or make decisions about an individual's health. Consult a qualified healthcare professional for medical advice.
