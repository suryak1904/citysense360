import streamlit as st
import numpy as np
import joblib
import cv2
import tempfile
import os

from tensorflow.keras.models import load_model

from services.cctv_service import CCTVService
from services.emergency_rag import EmergencyRAGService
from services.urban_planner import UrbanPlanner
from services.city_agent import CityOperationsAgent


st.set_page_config(
    page_title="CitySense360",
    layout="wide"
)

st.title("🏙️ CitySense360 — Smart City AI Control Panel")
st.markdown(
    "Unified AI platform for traffic, safety, environment, and city operations."
)


@st.cache_resource
def load_all_services():
    # ---- AQI ----
    aqi_model = load_model("models/aqi_regression_model.h5")
    aqi_scaler = joblib.load("models/aqi_scaler.pkl")

    # ---- Complaints ----
    complaint_model = joblib.load("models/complaint_classifier.pkl")
    complaint_vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

    # ---- Services ----
    cctv_service = CCTVService()
    emergency_rag = EmergencyRAGService()
    urban_planner = UrbanPlanner()
    city_agent = CityOperationsAgent()

    return (
        aqi_model,
        aqi_scaler,
        complaint_model,
        complaint_vectorizer,
        cctv_service,
        emergency_rag,
        urban_planner,
        city_agent,
    )


(
    aqi_model,
    aqi_scaler,
    complaint_model,
    complaint_vectorizer,
    cctv,
    emergency_rag,
    urban_planner,
    city_agent,
) = load_all_services()


module = st.sidebar.radio(
    "Select Module",
    [
        "📷 CCTV Surveillance",
        "🚨 Emergency Incident (RAG)",
        "🌫️ AQI Prediction",
        "📝 Complaint Analyzer",
        "🏗️ Generative Urban Planning",
        "🧠 City Operations Agent",
    ]
)


if module == "📷 CCTV Surveillance":
    st.header("📷 CCTV Surveillance System")

    mode = st.radio("Select Input Type", ["Image", "Video"])

    uploaded = st.file_uploader(
        f"Upload CCTV {mode}",
        type=["jpg", "png", "mp4"]
    )

    if uploaded:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(uploaded.read())
            input_path = tmp.name

        # ---------- IMAGE ----------
        if mode == "Image":
            st.subheader("🖼️ Image Analysis")

            annotated_img, event, saved_path = cctv.process_image(input_path)

            annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)

            st.image(
                annotated_img,
                caption="Annotated CCTV Image",
                use_column_width=True
            )
            st.json(event)

        # ---------- VIDEO ----------
        else:
            st.subheader("🎥 Video Analysis")

            output_video_path, event = cctv.process_video(input_path)

            st.video(output_video_path)
            st.json(event)


elif module == "🚨 Emergency Incident (RAG)":
    st.header("🚨 Emergency Incident Notification")

    incident_text = st.text_area(
        "Describe incident",
        height=180,
        placeholder="Example: CCTV detected vehicle collision with crowd gathering..."
    )

    if st.button("Generate Emergency Response"):
        if incident_text.strip():
            result = emergency_rag.handle_incident(incident_text)
            st.subheader("AI-Generated Response")
            st.write(result["response"])
        else:
            st.warning("Please enter incident details.")


elif module == "🌫️ AQI Prediction":
    st.header("🌫️ AQI Prediction")

    pm25 = st.number_input("PM2.5", min_value=0.0)
    pm10 = st.number_input("PM10", min_value=0.0)
    no2 = st.number_input("NO2", min_value=0.0)
    so2 = st.number_input("SO2", min_value=0.0)
    co = st.number_input("CO", min_value=0.0)
    o3 = st.number_input("O3", min_value=0.0)

    if st.button("Predict AQI"):
        X = np.array([[pm25, pm10, no2, so2, co, o3]])
        X_scaled = aqi_scaler.transform(X)
        prediction = aqi_model.predict(X_scaled)[0][0]

        st.success(f"Predicted AQI: {int(prediction)}")


elif module == "📝 Complaint Analyzer":
    st.header("📝 Citizen Complaint Analyzer")

    complaint_text = st.text_area(
        "Enter citizen complaint",
        height=160
    )

    if st.button("Classify Complaint"):
        if complaint_text.strip():
            vec = complaint_vectorizer.transform([complaint_text])
            category = complaint_model.predict(vec)[0]
            st.success(f"Assigned Category: {category}")
        else:
            st.warning("Please enter complaint text.")


elif module == "🏗️ Generative Urban Planning":
    st.header("🏗️ Generative Urban Planning")

    prompt = st.text_area(
        "Urban planning prompt",
        height=160,
        placeholder="Example: Smart city with green parks and efficient public transport"
    )

    if st.button("Generate City Layout"):
        if prompt.strip():
            image, _ = urban_planner.generate_plan(prompt, "urban_plan.png")
            st.image(image, use_column_width=True)
        else:
            st.warning("Please enter a prompt.")


elif module == "🧠 City Operations Agent":
    st.header("🧠 City Operations Agent")

    event_text = st.text_area(
        "Enter city event",
        height=180,
        placeholder="Example: High AQI detected and multiple respiratory complaints reported"
    )

    if st.button("Process City Event"):
        if event_text.strip():
            report = city_agent.handle_event(event_text)
            st.subheader("Agent Decision Report")
            st.json(report)
        else:
            st.warning("Please enter event details.")


st.markdown("---")
st.markdown(
    "**CitySense360** | AI-powered Smart City Intelligence Platform"
)
