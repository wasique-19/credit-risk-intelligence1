import requests
import streamlit as st

st.set_page_config(
    page_title="Credit Risk Intelligence",
    page_icon="💳",
    layout="wide",
)

st.title("💳 Credit Risk Intelligence")
st.subheader("Loan Default Risk Prediction Dashboard")

st.write(
    "Enter an applicant ID to analyze default probability, "
    "risk grade, business decision, and SHAP-based explanations."
)

applicant_id = st.number_input(
    "Applicant ID",
    min_value=1,
    step=1,
)

if st.button("Analyze Risk", type="primary"):
    try:
        response = requests.post(
            "http://127.0.0.1:8000/predict",
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

                st.divider()

                st.subheader("🔺 Positive Risk Contributors")

                for item in result["positive_contributors"]:
                    st.write(
                        f"**{item['feature']}** — "
                        f"{item['shap_value']:+.4f}"
                    )

                st.subheader("🔻 Negative Risk Contributors")

                for item in result["negative_contributors"]:
                    st.write(
                        f"**{item['feature']}** — "
                        f"{item['shap_value']:+.4f}"
                    )

    except requests.exceptions.RequestException:
        st.error(
            "Could not connect to the Credit Risk API. "
            "Please make sure FastAPI is running."
        )