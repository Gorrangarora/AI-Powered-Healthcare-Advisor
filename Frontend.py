"""
Healthcare AI - Optimized High Confidence Version
"""

import streamlit as st
from healthcare_ai_optimized import HealthcareAI
import os

st.set_page_config(page_title="Healthcare AI Pro", page_icon="🏥", layout="wide")

# Custom CSS
st.markdown("""
<style>
.big-title {
    font-size: 3.5rem;
    color: #1f77b4;
    text-align: center;
    font-weight: bold;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
}
.confidence-high {
    background: linear-gradient(135deg, #28a745, #20c997);
    color: white;
    padding: 1.5rem;
    border-radius: 15px;
    text-align: center;
    font-size: 2rem;
    font-weight: bold;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}
.emergency {
    background-color: #dc3545;
    color: white;
    padding: 2rem;
    border-radius: 10px;
    text-align: center;
    font-size: 1.5rem;
    font-weight: bold;
    animation: pulse 1.5s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.7; }
}
.severity-critical { 
    background: linear-gradient(135deg, #f8d7da, #f5c6cb);
    border-left: 5px solid #dc3545; 
    padding: 1rem;
    border-radius: 8px;
    margin: 1rem 0;
}
.severity-moderate { 
    background: linear-gradient(135deg, #fff3cd, #ffeaa7);
    border-left: 5px solid #ffc107; 
    padding: 1rem;
    border-radius: 8px;
    margin: 1rem 0;
}
.severity-normal { 
    background: linear-gradient(135deg, #d4edda, #c3e6cb);
    border-left: 5px solid #28a745; 
    padding: 1rem;
    border-radius: 8px;
    margin: 1rem 0;
}
.result-card {
    background: white;
    padding: 2rem;
    border-radius: 15px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# Initialize AI
@st.cache_resource
def load_ai():
    ai = HealthcareAI()
    if not ai.load():
        if os.path.exists('dataset_improved.csv'):
            with st.spinner("🔄 Training optimized model for HIGH CONFIDENCE... Please wait 2-3 minutes..."):
                ai.train()
        else:
            st.error("❌ dataset_improved.csv not found!")
            return None
    return ai

ai = load_ai()

# Header
st.markdown('<div class="big-title">🏥 Healthcare AI Pro</div>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center; font-size:1.3rem; color: #666;">🎯 Optimized for High Confidence Predictions</p>', unsafe_allow_html=True)
st.markdown("---")

if ai is None:
    st.stop()

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/medical-heart.png", width=90)
    
    st.markdown("### ⚡ High Confidence AI")
    st.success("""
    **NEW in v3.0:**
    - 🎯 Higher confidence scores!
    - 🚀 Better Accuracy
    - ⚡ Optimized model
    - ✅ Faster predictions
    """)
    
    st.markdown("### 📊 Model Stats")
    st.info("""
    - **Trees:** 200
    - **Features:** 1000
    - **Dataset:** 5580 entries
    """)
    
    st.markdown("### ⚠️ Emergency")
    st.error("""
    **Call Immediately:**
    - 🇮🇳 India: **108**
    - 🇺🇸 US: **911**
    - 🇬🇧 UK: **999**
    """)
    
    st.markdown("### ✅ Input Guide")
    st.info("""
    **Good Examples:**
    - fever, dry cough, fatigue
    - severe headache, nausea
    - runny nose, sneezing
    
    **Minimum:** 3-4 symptoms
    """)

# Main content
st.markdown("### 💬 Describe Your Symptoms")

col1, col2 = st.columns([3, 1])

with col1:
    symptoms = st.text_area(
        "Enter your symptoms:",
        height=130,
        placeholder="Example: fever, dry cough, body ache, fatigue, headache",
        help="Enter at least 2-3 symptoms. Be specific for best results!"
    )

with col2:
    st.markdown("#### 🎯 Try These")
    
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🤒 Flu", use_container_width=True):
            symptoms = "fever, dry cough, body ache, fatigue, headache"
            st.rerun()
        if st.button("🤧 Cold", use_container_width=True):
            symptoms = "runny nose, sneezing, sore throat, nasal congestion"
            st.rerun()
    with col_b:
        if st.button("🤕 Migraine", use_container_width=True):
            symptoms = "severe headache, nausea, sensitivity to light, throbbing pain"
            st.rerun()
        if st.button("🚨 Emergency", use_container_width=True):
            symptoms = "severe chest pain, difficulty breathing, arm pain, sweating"
            st.rerun()

# Predict button
if st.button("🔍 Analyze Symptoms", type="primary", use_container_width=True):
    if not symptoms or len(symptoms.strip()) < 3:
        st.warning("⚠️ Please enter your symptoms")
    else:
        with st.spinner("🔄 Analyzing with optimized AI model..."):
            result = ai.predict(symptoms)
        
        # Handle invalid input
        if not result.get('is_valid', True):
            st.error(f"### {result['error']}")
            
            st.markdown("---")
            st.markdown("### ✅ Valid Examples:")
            
            col1, col2 = st.columns(2)
            with col1:
                st.success("""
                **Good:**
                - ✓ fever, cough, headache
                - ✓ severe headache, nausea
                - ✓ runny nose, sneezing, sore throat
                """)
            with col2:
                st.error("""
                **Bad:**
                - ✗ vomit (too short)
                - ✗ headache (only 1)
                - ✗ random text
                """)
            
            st.info("💡 **Tip:** Use specific medical terms and provide 2-3 symptoms minimum")
            st.stop()
        
        if result['is_emergency']:
            # Emergency alert
            st.markdown(f'<div class="emergency">{result["message"]}</div>', unsafe_allow_html=True)
            
            st.markdown("---")
            st.error("### 🚨 CALL EMERGENCY SERVICES IMMEDIATELY!")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("""
                <div style='text-align: center; background: #dc3545; color: white; padding: 1rem; border-radius: 10px;'>
                <h3>🇮🇳 India</h3>
                <h1>108</h1>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown("""
                <div style='text-align: center; background: #dc3545; color: white; padding: 1rem; border-radius: 10px;'>
                <h3>🇺🇸 US</h3>
                <h1>911</h1>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                st.markdown("""
                <div style='text-align: center; background: #dc3545; color: white; padding: 1rem; border-radius: 10px;'>
                <h3>🇬🇧 UK</h3>
                <h1>999</h1>
                </div>
                """, unsafe_allow_html=True)
            
        else:
            # Success message
            st.balloons()
            st.success("✅ Analysis Complete - High Confidence Prediction!")
            
            st.markdown("---")
            
            # Main result card
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"## 🏥 {result['disease']}")
                st.markdown(f"**{result['description']}**")
            
            with col2:
                # Confidence display with gradient
                confidence = result['confidence']
                if confidence >= 80:
                    conf_color = "#28a745"
                    conf_emoji = "🎯"
                    conf_text = "Excellent"
                elif confidence >= 60:
                    conf_color = "#20c997"
                    conf_emoji = "✅"
                    conf_text = "High"
                elif confidence >= 40:
                    conf_color = "#ffc107"
                    conf_emoji = "⚠️"
                    conf_text = "Good"
                else:
                    conf_color = "#fd7e14"
                    conf_emoji = "ℹ️"
                    conf_text = "Moderate"
                
                st.markdown(f"""
                <div style='background: {conf_color}; color: white; padding: 1.5rem; border-radius: 15px; text-align: center;'>
                    <div style='font-size: 3rem;'>{conf_emoji}</div>
                    <div style='font-size: 2.5rem; font-weight: bold;'>{confidence}%</div>
                    <div style='font-size: 1.2rem;'>{conf_text} Confidence</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Severity badge
            st.markdown("---")
            severity_class = f"severity-{result['severity'].lower()}"
            st.markdown(f'<div class="{severity_class}"><h3>⚠️ Severity Level: {result["severity"]}</h3></div>', unsafe_allow_html=True)
            
            # Tabs for information
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "🏠 Home Care", 
                "🌿 Natural", 
                "💊 Medicines",
                "🛡️ Prevention",
                "👨‍⚕️ Doctor Advice"
            ])
            
            with tab1:
                st.markdown("### 🏠 Home Remedies")
                for i, remedy in enumerate(result['home_remedies'], 1):
                    st.markdown(f"**{i}.** {remedy}")
            
            with tab2:
                st.markdown("### 🌿 Natural Remedies")
                for i, remedy in enumerate(result['natural_remedies'], 1):
                    st.markdown(f"**{i}.** {remedy}")
                st.caption("⚠️ Always consult a healthcare provider before taking supplements")
            
            with tab3:
                st.markdown("### 💊 Over-the-Counter Medicines")
                st.warning("⚠️ **NO DOSAGE PROVIDED** - Consult pharmacist or doctor")
                for i, medicine in enumerate(result['otc_medicines'], 1):
                    st.markdown(f"**{i}.** {medicine}")
                st.caption("📋 Always read medicine labels and follow instructions carefully")
            
            with tab4:
                st.markdown("### 🛡️ Prevention Tips")
                for i, tip in enumerate(result['prevention'], 1):
                    st.markdown(f"**{i}.** {tip}")
            
            with tab5:
                st.markdown("### 👨‍⚕️ When to See a Doctor")
                
                if result['severity'] == 'Critical':
                    st.error("""
                    ### ⚠️ CRITICAL - Immediate Medical Attention Required
                    
                    **Please consult a doctor immediately because:**
                    - This is a serious condition
                    - Requires professional medical care
                    - May need prescription medication
                    - Should not be self-treated
                    
                    **Action:** Schedule appointment TODAY or visit emergency room
                    """)
                elif result['severity'] == 'Moderate':
                    st.warning("""
                    ### ⚠️ MODERATE - Medical Consultation Recommended
                    
                    **See a doctor if:**
                    - Symptoms persist beyond 2-3 days
                    - Symptoms worsen instead of improving
                    - New symptoms develop
                    - You feel very unwell
                    
                    **Action:** Schedule doctor appointment within 2-3 days
                    """)
                else:
                    st.info("""
                    ### ℹ️ NORMAL - Home Care Usually Sufficient
                    
                    **Self-care is appropriate, but see doctor if:**
                    - Symptoms last longer than 5-7 days
                    - Symptoms significantly worsen
                    - You develop fever or severe pain
                    - You have any concerns
                    
                    **Action:** Monitor symptoms, seek help if needed
                    """)
            
            # Additional monitoring advice
            st.markdown("---")
            st.markdown("### 📌 Important Reminders")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.info("""
                **💧 Stay Hydrated**
                - Drink 8-10 glasses water
                - Clear fluids
                - Avoid alcohol/caffeine
                """)
            with col2:
                st.info("""
                **😴 Rest Well**
                - Get adequate sleep
                - Avoid strenuous activity
                - Listen to your body
                """)
            with col3:
                st.info("""
                **📊 Monitor Symptoms**
                - Track changes
                - Note improvements
                - Record new symptoms
                """)
            
            # Disclaimer
            st.markdown("---")
            st.warning("""
            ### ⚠️ MEDICAL DISCLAIMER
            
            This is an AI-powered preliminary health guidance tool. **This is NOT a medical diagnosis** 
            and should not replace professional medical advice, diagnosis, or treatment. 
            
            **Always consult a qualified healthcare provider** for accurate medical assessment. 
            If you have serious health concerns or symptoms worsen, please see a doctor immediately.
            
            This tool is for educational and informational purposes only.
            """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem; background: #f8f9fa; border-radius: 10px;">
    <h3 style="color: #1f77b4;">🏥 Healthcare AI Pro - v3.0 Optimized</h3>
    <p><strong>High Confidence Predictions |5580 Dataset Entries</strong></p>
    <p>Optimized Random Forest Model</p>
    <p>⚕️ For educational purposes only • Not a replacement for medical professionals</p>
    <p style="font-size: 0.9rem; color: #999;">Made with using Python, Scikit-learn & Streamlit</p>
</div>
""", unsafe_allow_html=True)
