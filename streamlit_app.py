from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from boston_house_price_predictor.config import MODEL_PATH
from boston_house_price_predictor.inference import MissingArtifactError, predict_price


st.set_page_config(page_title="Boston House Price Predictor", layout="wide")

st.markdown(
    """
    <style>
    .hero {
      border-radius: 14px;
      padding: 1.1rem 1.3rem;
      background: linear-gradient(90deg, #F9D976 0%, #F39F86 100%);
      color: #1D1D1D;
      margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h2 style="margin:0;">Boston House Price Predictor</h2>
      <p style="margin:0.4rem 0 0 0;">Estimate the median value of a Boston home from 13 neighbourhood and property features.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not MODEL_PATH.exists():
    st.warning("Model artifact is missing. Train it first with: python scripts/train_model.py")
    if st.button("Run training now"):
        result = subprocess.run(
            [sys.executable, "scripts/train_model.py"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            st.success("Model trained successfully. Reload to begin predictions.")
        else:
            st.error("Training failed.")
            st.code(result.stderr or result.stdout)

left, right = st.columns(2)

with left:
    st.subheader("Property and Location Inputs")
    crim = st.number_input("CRIM", value=0.10, help="Per capita crime rate by town")
    zn = st.number_input("ZN", value=0.0, help="Residential land zoned proportion")
    indus = st.number_input("INDUS", value=8.0, help="Non-retail business acres proportion")
    chas = st.selectbox("CHAS", [0, 1], help="1 if tract bounds Charles River")
    nox = st.number_input("NOX", value=0.50, help="Nitric oxides concentration")
    rm = st.number_input("RM", value=6.2, help="Average rooms per dwelling")
    age = st.number_input("AGE", value=65.0, help="Proportion of older owner-occupied units")

with right:
    st.subheader("Accessibility and Socio-economic Inputs")
    dis = st.number_input("DIS", value=4.0, help="Weighted distance to employment centers")
    rad = st.number_input("RAD", value=4.0, help="Accessibility index to radial highways")
    tax = st.number_input("TAX", value=300.0, help="Property-tax rate per $10,000")
    ptratio = st.number_input("PTRATIO", value=18.0, help="Pupil-teacher ratio")
    b = st.number_input("B", value=390.0, help="1000(Bk - 0.63)^2")
    lstat = st.number_input("LSTAT", value=12.0, help="Percentage lower status population")

if st.button("Predict Price", type="primary"):
    payload = {
        "CRIM": float(crim),
        "ZN": float(zn),
        "INDUS": float(indus),
        "CHAS": int(chas),
        "NOX": float(nox),
        "RM": float(rm),
        "AGE": float(age),
        "DIS": float(dis),
        "RAD": float(rad),
        "TAX": float(tax),
        "PTRATIO": float(ptratio),
        "B": float(b),
        "LSTAT": float(lstat),
    }

    try:
        estimate = predict_price(payload)
        st.success(f"Estimated house price: ${estimate * 1000:,.2f}")
        st.caption("Prediction unit: USD (dataset target MEDV is in $1000s).")
    except MissingArtifactError as error:
        st.error(str(error))
    except Exception as error:
        st.error(f"Prediction failed: {error}")
