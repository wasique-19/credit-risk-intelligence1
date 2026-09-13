import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(
    page_title="Credit Risk Intelligence",
    page_icon="💳",
    layout="wide",
)

st.title("💳 Credit Risk Intelligence")
st.subheader("Loan Default Risk Prediction Dashboard")

st.write(
    "Analyze an applicant's predicted default probability, "
    "risk grade, business decision, and SHAP-based risk drivers."
)

st.divider()

applicant_id = st.number_input(
    "Applicant ID",
    min_value=1,
    step=1,
)

if st.button("Analyze Risk", type="primary"):

    try:
        response = requests.post(
            API_URL,
            json={"applicant_id": int(applicant_id)},
            timeout=30,
        )

        if response.status_code != 200:
            st.error(f"API error: {response.status_code}")

        else:
            result = response.json()

            if "error" in result:
                st.error(result["error"])

            else:
                probability = result["default_probability"]
                risk_grade = result["risk_grade"]
                decision = result["decision"]

                st.success("Risk analysis completed.")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        "Default Probability",
                        f"{probability:.2%}",
                    )

                with col2:
                    st.metric(
                        "Risk Grade",
                        risk_grade,
                    )

                with col3:
                    st.metric(
                        "Decision",
                        decision,
                    )

                st.progress(
                    min(probability, 1.0),
                    text=f"Predicted default risk: {probability:.2%}",
                )

                st.divider()

                st.subheader("Risk Interpretation")

                if probability < 0.05:
                    st.info(
                        "Low predicted default risk. "
                        "Current decision rule: APPROVE."
                    )
                elif probability < 0.20:
                    st.warning(
                        "Moderate predicted default risk. "
                        "Current decision rule: MANUAL REVIEW."
                    )
                else:
                    st.error(
                        "High predicted default risk. "
                        "Current decision rule: REJECT."
                    )

                st.divider()

                positive_col, negative_col = st.columns(2)

                with positive_col:
                    st.subheader("🔺 Positive Risk Contributors")

                    for item in result["positive_contributors"]:
                        st.write(
                            f"**{item['feature']}**  \n"
                            f"SHAP: `{item['shap_value']:+.4f}`"
                        )

                with negative_col:
                    st.subheader("🔻 Negative Risk Contributors")

                    for item in result["negative_contributors"]:
                        st.write(
                            f"**{item['feature']}**  \n"
                            f"SHAP: `{item['shap_value']:+.4f}`"
                        )

                st.caption(
                    "SHAP values explain model behavior and should not "
                    "be interpreted as causal effects."
                )

    except requests.exceptions.RequestException:
        st.error(
            "Could not connect to the Credit Risk API. "
            "Please make sure FastAPI is running."
        )