# 🏥 AI-Powered Healthcare Advisor (AIHCA)

An AI-based healthcare advisory system that analyzes user-entered symptoms, predicts possible diseases, provides home remedies, natural suggestions, OTC guidance, and identifies critical conditions requiring immediate medical attention.

---

## 📌 Project Overview

AIHCA is designed to provide accessible, affordable, and preliminary healthcare guidance using Machine Learning and Natural Language Processing.

The system helps users:

- Understand possible diseases based on symptoms  
- Receive home and natural remedy suggestions  
- Get safe OTC medication guidance (no dosage provided)  
- Detect emergency symptoms and recommend immediate doctor consultation  

⚠️ This system does NOT replace professional medical advice.

---

## 🎯 Key Features

- Symptom-to-Disease Prediction  
- Severity Classification (Normal / Moderate / Critical)  
- Emergency Override System  
- Home Remedies & Natural Cure Suggestions  
- OTC Medicine Recommendations (No Prescription Dosage)  
- Preventive Healthcare Advice  
- User-Friendly Web Interface  

---

## 🧠 System Architecture

User Input (Symptoms)  
→ Text Preprocessing  
→ TF-IDF Vectorization  
→ Random Forest Classifier  
→ Severity Detection (Rule-Based)  
→ Advice Engine  
→ Output with Disclaimer  

---

## 🛠️ Tech Stack

- Python  
- Scikit-learn  
- Pandas  
- NumPy  
- Streamlit (Frontend)  
- Joblib (Model Saving)  

---

## 📂 Project Structure

```
healthcare_ai/
│
├── data/
│   └── dataset.csv
│
├── models/
│   └── disease_model.pkl
│
├── train_model.py
├── app.py
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run the Project

### 1️⃣ Clone the Repository

```
git clone https://github.com/your-username/AIHCA.git
cd AIHCA
```

---

### 2️⃣ Install Dependencies

```
pip install -r requirements.txt
```

---

### 3️⃣ Train the Model

```
python train_model.py
```

This will:
- Load dataset  
- Train TF-IDF + Random Forest model  
- Save trained model inside the models folder  

---

### 4️⃣ Run the Application

```
streamlit run app.py
```

The application will open in your browser.

---

## 📊 Model Details

- Feature Extraction: TF-IDF Vectorizer  
- Classifier: Random Forest (Multi-Class Classification)  
- Evaluation Metrics:
  - Accuracy  
  - Precision  
  - Recall  
  - F1 Score  

---

## 🚨 Emergency Detection Logic

The system overrides ML prediction if critical symptoms are detected such as:

- Chest pain  
- Breathlessness  
- Unconsciousness  
- Severe bleeding  
- Stroke indicators  

In such cases, it displays:

**CRITICAL CONDITION – SEEK IMMEDIATE MEDICAL ATTENTION**

---

## 🔮 Future Enhancements

- Multilingual Support  
- AI-Based Image Diagnosis  
- Voice Interaction  
- Wearable Device Integration  
- Personalized Health Tracking  

---

## ⚠️ Disclaimer

This system provides AI-generated preliminary health guidance and is not a substitute for professional medical diagnosis or treatment. Always consult a qualified healthcare provider for serious or persistent symptoms.

---

## 👨‍💻 Author

Gorrang Arora  
B.Tech Computer Science & Engineering  
Manipal University Jaipur  
Academic Year: 2025–2026  
