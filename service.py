import streamlit as st
import pickle
import random
from sklearn.ensemble import RandomForestClassifier
from rdkit import Chem
from rdkit.Chem import PandasTools, Descriptors
import numpy as np


def calculate_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Извлекаем дескрипторы
    features = {}
    for i, j in Descriptors.descList:
        features[i] = j(mol)

    return features


THRESHOLD = 0.28
COLUMNS = ['PEOE_VSA10',
           'NumHDonors',
           'BCUT2D_MRHI',
           'SMR_VSA3',
           'BCUT2D_MWHI',
           'MaxPartialCharge',
           'SMR_VSA5',
           'MaxEStateIndex',
           'MinAbsEStateIndex',
           'PEOE_VSA3',
           'BCUT2D_MRLOW',
           'BalabanJ',
           'FpDensityMorgan3',
           'BCUT2D_CHGLO',
           'BCUT2D_LOGPLOW',
           'SMR_VSA4',
           'fr_aryl_methyl',
           'SlogP_VSA4',
           'SlogP_VSA8',
           'PEOE_VSA4',
           'SMR_VSA9',
           'BCUT2D_MWLOW',
           'SlogP_VSA12',
           'VSA_EState6',
           'VSA_EState10',
           'fr_azide',
           'fr_SH'
           ]


@st.cache_resource
def load_model(path):
    with open(path, 'rb') as f:
        model = pickle.load(f)
        return model


model_path = "D:\\pythonProject\\model_rfc.pkl"
model = load_model(model_path)

st.title("HIV GenAI")

st.subheader("Choose your mode")
mode = st.radio("", ["Classification", "Generation"])

if mode == "Classification":
    smiles_input = st.text_input("Enter your SMILES string:")

    if st.button("Classify"):
        if smiles_input:
            try:
                features = calculate_descriptors(smiles_input)

                if features is None:
                    st.error("Invalid SMILES string!")
                else:
                    feature_vector = [features.get(col, 0) for col in COLUMNS]
                    features_array = np.array(feature_vector).reshape(1, -1)

                    probs = model.predict_proba(features_array)[:, 1]
                    prediction = "Active" if probs[0] >= THRESHOLD else "Inactive"

                    st.success(f"Result: {prediction} (Probability: {probs[0]:.2f})")
            except Exception as e:
                st.error(f"Error processing input: {e}")
        else:
            st.warning("Please enter a SMILES string.")

elif mode == "Generation":
    if st.button("Generate"):
        generated_molecule = "C1=CC=CC=C1"
        st.success(f"Generated SMILES: {generated_molecule}")
