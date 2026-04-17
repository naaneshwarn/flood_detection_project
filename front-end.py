import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from datetime import datetime

# 1. Page Configuration
st.set_page_config(
    page_title="FloodGuard India",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)


def apply_custom_styles(color_code):
    st.markdown(f"""
        <style>
        /* Main Background */
        .stApp {{
            background-color: {color_code}05; 
        }}
        
        /* Prediction Card */
        .result-card {{
            background: white;
            padding: 30px;
            border-radius: 20px;
            border-top: 8px solid {color_code};
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
            margin-bottom: 25px;
        }}

        /* Emergency Action Card Styles */
        .action-card {{
            background-color: white;
            border-left: 5px solid {color_code};
            padding: 15px;
            margin-bottom: 10px;
            border-radius: 8px;
            box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
            display: flex;
            align-items: center;
        }}
        .action-text {{
            color: #333;
            font-weight: 500;
            margin-left: 10px;
        }}
        </style>
    """, unsafe_allow_html=True)


st.sidebar.markdown("## 🛰️ Control Center")
with st.sidebar:
    input_date = st.date_input("Forecast Date", datetime.now())
    city_name = st.text_input("City Name", "Chennai")
    
    try:
        states_res = requests.get("http://127.0.0.1:8000/states").json()
        state_list = states_res['states']
    except:
        state_list = ["Tamil Nadu", "Kerala", "Assam", "Bihar", "Maharashtra", "Karnataka"]
    
    selected_state = st.selectbox("State", state_list)
    duration = st.slider("Duration (Days)", 1, 30, 3)
    cause = st.selectbox("Primary Cause", ["Heavy Rains", "Flash Flood", "Cyclone", "Dam Breach", "Landslide"])
    
    predict_btn = st.button("🚀 ANALYZE RISK", use_container_width=True)


if predict_btn:
    payload = {
        "date": str(input_date),
        "city": city_name,
        "state": selected_state,
        "duration_days": duration,
        "cause": cause
    }

    try:
        response = requests.post("http://127.0.0.1:8000/predict", json=payload)
        data = response.json()
        
        apply_custom_styles(data['color'])

        # Dynamic Result Header
        st.markdown(f"""
            <div class="result-card">
                <h1 style='color: {data['color']}; margin-top: 0;'>{data['emoji']} {data['risk_level']} Severity Forecast</h1>
                <h4 style='color: #666;'>Target: {data['city']}, {selected_state}</h4>
                <hr style='border: 0.5px solid #eee;'>
                <p style='font-size: 1.1rem;'><b>Insight:</b> {data['advice']}</p>
            </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns([1, 1], gap="large")
        
        with col1:
            st.markdown("### 📊 Analysis Metrics")
            m1, m2 = st.columns(2)
            m1.metric("Confidence", f"{data['confidence']}%")
            m2.metric("Duration", f"{duration} Days")
            
          
            st.write("#### Risk Probability Distribution")
            prob_df = pd.DataFrame(data['probabilities'].items(), columns=['Risk', 'Prob'])
            fig = px.pie(prob_df, values='Prob', names='Risk', hole=.5,
                         color='Risk',
                         color_discrete_map={'Low':'#22c55e', 'Medium':'#f59e0b', 'High':"#983a3a", 'Extreme':"#ff0000"})
            fig.update_layout(showlegend=False, margin=dict(t=0, b=0, l=0, r=0))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("### 🚨 Emergency Action Plan")
            # Loop through actions and display as custom cards
            for action in data['recommended_actions']:
                st.markdown(f"""
                    <div class="action-card">
                        <span style='font-size: 1.2rem;'>✅</span>
                        <span class="action-text">{action}</span>
                    </div>
                """, unsafe_allow_html=True)

         
            st.write("#### Predicted Intensity Trend")
            chart_data = pd.DataFrame({
                'Timeline': [f'Day {i}' for i in range(1, duration + 1)],
                'Risk Score': [(i * (data['confidence']/10)) for i in range(1, duration + 1)]
            })
            st.line_chart(chart_data.set_index('Timeline'), color=data['color'])

    except Exception as e:
        st.error(f"Backend Offline! Error: {e}")
else:
    # Initial Welcome Screen
    st.title("Welcome to FloodGuard India")
    st.info("Please fill the simulation details on the left and click 'Analyze Risk'.")
   