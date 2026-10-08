# Network Intrusion Detection with Random Forest (UNSW-NB15)

CST9 Final Project · University of Mindanao, College of Computing Education

Streamlit app that classifies network flows as **Normal** or **Attack** using a Random Forest trained on UNSW-NB15.

## Repository layout
```
app.py                  Streamlit application
requirements.txt        Dependencies (pin scikit-learn to the training version)
model/                  random_forest_final.pkl, preprocessor_final.pkl,
                        feature_selector_final.pkl, app_meta.json, sample_input.csv
notebooks/              CST9_Final_Project_clean.ipynb (training and evaluation)
```

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deployment
Deployed on Streamlit Community Cloud from this repository (main file: `app.py`).

## Results (official test partition)
See the "About the model" tab in the app, or Section 12 of the notebook.
