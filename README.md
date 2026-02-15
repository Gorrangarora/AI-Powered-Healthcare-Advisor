# AI-Powered Healthcare Advisor (AIHCA) (http://10.174.6.249:8501)

AIHCA is a machine learning-based healthcare assistance system that helps users understand possible diseases based on the symptoms they enter. The system provides preliminary guidance such as home remedies, natural suggestions, OTC medicine recommendations (without dosage), and alerts users in case of critical conditions.

This project is developed as part of a B.Tech Computer Science academic project.

---

## About the Project

Many people rely on random internet sources when they feel sick, which can lead to confusion, anxiety, and sometimes wrong self-treatment. Hospital visits can also involve long waiting hours and high consultation costs, especially for minor issues.

AIHCA is built to act as a first-level healthcare support system. It does not replace a doctor but helps users understand:

- What disease might be associated with their symptoms  
- Whether the condition seems normal, moderate, or critical  
- Basic home and natural remedies  
- When it is necessary to consult a doctor immediately  

The system is especially helpful for elderly individuals and people who may not easily access hospitals.

---

## Main Features

- Symptom-based disease prediction  
- Severity classification (Normal / Moderate / Critical)  
- Emergency symptom detection override  
- Home remedy suggestions  
- Natural treatment advice  
- Over-the-counter (OTC) medicine suggestions (no dosage provided)  
- Clear medical disclaimer  

---

## How the System Works

1. The user enters symptoms in text form.
2. The system processes and cleans the text.
3. TF-IDF converts the text into numerical features.
4. A Random Forest model predicts the most likely disease.
5. A rule-based module checks for emergency keywords.
6. The system displays:
   - Predicted disease  
   - Severity level  
   - Remedies and suggestions  
   - Doctor recommendation if needed  

---

## Technologies Used

- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- Streamlit  
- Joblib  

---

## Project Structure

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

## How to Run the Project

### Step 1: Clone the Repository

```
git clone https://github.com/your-username/AIHCA.git
cd AIHCA
```

### Step 2: Install Required Libraries

```
pip install -r requirements.txt
```

### Step 3: Train the Model

```
python train_model.py
```

### Step 4: Run the Application

```
streamlit run app.py
```

The application will open in your browser.

---

## Model Information

- Text Feature Extraction: TF-IDF  
- Classification Algorithm: Random Forest  
- Type: Multi-class classification  
- Evaluation Metrics: Accuracy, Precision, Recall, F1 Score  

---

## Emergency Handling

If the user enters critical symptoms such as:

- Chest pain  
- Breathlessness  
- Unconsciousness  
- Severe bleeding  

The system will immediately display a warning message advising the user to seek medical help.

---

## Future Improvements

- Multilingual support for regional languages  
- Image-based diagnosis using deep learning  
- Voice interaction support  
- Integration with wearable health devices  

---

## Disclaimer

This system is developed for academic and informational purposes only. It provides preliminary guidance and does not replace professional medical advice. Users are strongly advised to consult a qualified healthcare professional for proper diagnosis and treatment.

---

## Author

Gorrang Arora  
B.Tech Computer Science & Engineering  
Manipal University Jaipur  
Academic Year: 2025–2026
