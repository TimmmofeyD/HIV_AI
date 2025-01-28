import streamlit as st
import random

st.title("HIV GenAI")

st.subheader("Choose your mode")
mode = st.radio("", ["Classification", "Generation"])

if mode == "Classification":
    smiles_input = st.text_input("Enter your SMILES string:")
    if st.button("Classify"):
        if smiles_input:
            # model
            result = random.choice(["Active", "Inactive"])
            st.success(f"Result: {result}")
        else:
            st.warning("Please enter a SMILES string.")

elif mode == "Generation":
    if st.button("Generate"):
        generated_molecule = "C1=CC=CC=C1"
        st.success(f"Generated SMILES: {generated_molecule}")
