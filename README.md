# 🏙️ CitySense360

### AI-Powered Smart City Intelligence & Public Infrastructure Automation

---

##  Project Overview

**CitySense360** is a unified **AI-driven smart city platform** that demonstrates how **Deep Learning, Computer Vision, NLP, Generative AI, RAG, and Agentic AI** can be integrated to solve real-world urban challenges.

This project is developed as a **learning-oriented, production-style system** for academic evaluation, focusing on **modular design, model integration, and end-to-end AI workflows** rather than live city deployment.

---

##  Key Objectives

* Predict traffic congestion using time-series models
* Detect vehicles, people, accidents, crowds via CCTV analysis
* Identify road infrastructure damage
* Predict air quality (AQI) and energy consumption
* Analyze citizen complaints using NLP
* Generate emergency response guidance using RAG + LLM
* Produce urban planning visualizations using generative models
* Orchestrate city operations using Agentic AI
* Provide a unified Streamlit-based dashboard

---

##  AI Modules Implemented

###  Module 1 — Traffic Congestion Prediction

* Model: **LSTM**
* Input: Historical traffic volume
* Output: Future congestion levels

---

###  Module 2 — Road Damage Detection

* Model: **Faster R-CNN**
* Dataset: RDD2022
* Detects: potholes, cracks, surface damage

---

###  Module 3 — Citizen Feedback & Complaint Analyzer

* Model: **TF-IDF + Machine Learning Classifier**
* Input: Complaint text
* Output: Department / category classification

---

###  Module 4 — AI-Powered CCTV Surveillance

* Model: **YOLOv8**
* Detects:

  * Vehicles & people
  * Crowd gathering
  * Overspeeding (logic-based)
  * Illegal parking
  * Possible accidents
* Output: Annotated images/videos + alerts

---

###  Module 5 — Pollution & AQI Prediction

* Model: **Deep Neural Network**
* Input: Pollution sensor values
* Output: AQI prediction

---

###  Module 6 — Emergency Incident Notification System

* Technique: **RAG (Retrieval Augmented Generation)**
* LLM: **Local Qwen via Ollama**
* Knowledge base: Emergency SOP documents
* Output: Actionable emergency response guidance

---

###  Module 7 — Smart Energy Usage Monitoring

* Model: **GRU**
* Input: Energy consumption time-series
* Output: Future power usage prediction

---

### Module 8 — Public Transport Demand Forecasting

* Reuses traffic time-series logic
* Model: **LSTM**
* Output: Passenger demand trends

---

###  Module 9 — Generative Urban Planning

* Model: **Diffusion Models**
* Input: Text prompt
* Output: Generated city layout / planning visualization

---

###  Module 10 — AI Agent for City Operations

* Type: **Rule-based Agentic AI**
* Function:

  * Classifies events
  * Routes tasks to relevant modules
  * Generates consolidated reports

---

##  System Architecture (High Level)

```
User (Streamlit UI)
        ↓
   app.py (Controller)
        ↓
   services/
        ↓
   Saved Models (models/)
        ↓
 Predictions / Images / Reports
```

---

##  Project Structure

```
CITYSENSE360/
│
├── app.py                     # Streamlit UI entry point
├── requirements.txt
│
├── data/
│   ├── traffic/
│   ├── cctv/
│   ├── pollution/
│   ├── energy/
│   ├── complaints/
│   ├── emergency_docs/
│
├── models/
│   ├── traffic_lstm.h5
│   ├── road_damage_fasterrcnn.pth
│   ├── aqi_regression_model.h5
│   ├── energy_gru_model.h5
│   ├── complaint_classifier.pkl
│   ├── tfidf_vectorizer.pkl
│
├── services/
│   ├── cctv_service.py
│   ├── emergency_rag.py
│   ├── urban_planner.py
│   ├── city_agent.py
│
├── outputs/
│   ├── annotated_images/
│   ├── annotated_videos/
│
├── vectorstore/
│   └── emergency_faiss/
│
└── notebooks/
    └── training_notebooks.ipynb
```

---

## Tech Stack

* **Programming:** Python 3.11
* **Deep Learning:** TensorFlow, Keras, PyTorch
* **Computer Vision:** OpenCV, YOLOv8, Faster R-CNN
* **NLP:** TF-IDF, LangChain
* **RAG:** FAISS + Local LLM (Qwen via Ollama)
* **Generative AI:** Diffusers
* **UI:** Streamlit
* **Deployment:** Local (CPU), modular & extensible

---

##  How to Run the Project

###  Create Environment (Recommended)

```bash
conda create -n citysense360 python=3.11 -y
conda activate citysense360
```

---

### Install Dependencies

```bash
pip install -r requirements.txt
pip install langchain-text-splitters
```

---

### 3️Run the Application

```bash
streamlit run app.py
```

Open in browser:

```
http://localhost:8501
```

---

## Model Usage

* Models are **pre-trained** and stored in `/models`
* The system performs **inference only**
* Live APIs are **not mandatory** for evaluation
* Modular services allow future live data integration

---

##  Evaluation Metrics Used

* **Classification:** Accuracy, Precision, Recall
* **Regression:** MAE, MSE
* **Object Detection:** IoU, visual verification
* **System Metrics:** Latency, modularity, reliability

---

##  Notes

* TensorFlow warnings (oneDNN, deprecated APIs) are **expected and harmless**
* Ollama runs as a **local service**, not a Python dependency
* This project emphasizes **architecture and integration**, not production deployment
Excluded Items:
Trained deep learning models (.h5, .pth, .pkl)
---
<img width="1903" height="920" alt="image" src="https://github.com/user-attachments/assets/58a9078a-e4fc-431e-a4e7-22d7461d185b" />



##  Author

**Surya K**
AI & Data Science Practitioner

