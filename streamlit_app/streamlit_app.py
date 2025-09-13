
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

# Import components directly
from components.map_renderer import MapRenderer, MapType
from components.charts import ChartRenderer
from components.recommendation_card import RecommendationCard

# Must be the first Streamlit command
st.set_page_config(
    page_title="UHI Mitigation Planner",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state
def init_session_state():
    """Initialize session state variables."""
    if 'sample_data' not in st.session_state:
        # Create sample data for demo
        st.session_state.sample_data = pd.DataFrame({
            "lat": [18.52, 18.53, 18.54, 18.55, 18.56],
            "lon": [73.85, 73.86, 73.87, 73.88, 73.89],
            "neighborhood": ["shivaji_nagar", "kothrud", "yerwada", "aundh", "koregaon_park"],
            "hvi": [0.35, 0.62, 0.81, 0.58, 0.75],
            "lst": [34.8, 37.1, 41.3, 35.5, 36.8],
            "population": [12000, 18000, 24000, 13000, 15000],
            "elderly_ratio": [0.08, 0.11, 0.05, 0.09, 0.10],
            "low_income_ratio": [0.22, 0.31, 0.15, 0.25, 0.18],
            "ac_prevalence": [0.15, 0.3, 0.45, 0.38, 0.42],
        })
    
    if 'sample_rec' not in st.session_state:
        st.session_state.sample_rec = {
            "neighborhood": "shivaji_nagar",
            "priority_score": 0.72,
            "hvi": 0.81,
            "avg_temperature": 41.3,
            "population": 24000,
            "estimated_cost": 1500000,
            "roi": 18.5,
            "payback_period": 3.2,
            "recommended_actions": ["cool_roofs", "urban_greening"],
            "implementation_time": 12,
            "implementation_plan": {
                "actions": [
                    {"action": "cool_roofs", "co_benefits": ["energy_savings", "improved_comfort"]},
                    {"action": "urban_greening", "co_benefits": ["air_quality", "biodiversity"]}
                ]
            },
        }

# Initialize session state
init_session_state()

# Load local CSS if exists
def local_css(file_name: str):
    """Load local CSS file if present."""
    try:
        css_path = Path(__file__).resolve().parent / file_name
        if css_path.exists():
            with open(css_path, "r", encoding="utf-8") as f:
                st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error loading CSS: {e}")

local_css("style.css")

# Main page content
st.title("🌡️ Urban Heat Island Mitigation Planner")
st.markdown("### Climate-Resilient Urban Planning for Indian Cities")

# Key metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Cities Analyzed", "2", "Pune & Nashik")
with col2:
    st.metric("Avg. UHI Intensity", "5.2°C", "+0.8°C from 2020")
with col3:
    st.metric("Population at Risk", "3.2M", "45% of total")
with col4:
    st.metric("Mitigation Potential", "3.5°C", "in 5 years")

st.markdown("---")

# City selection
col1, col2 = st.columns([2, 1])
with col1:
    st.subheader("Select City for Analysis")
    city = st.selectbox("Choose a city", ["Pune", "Nashik"], key="overview_city")
with col2:
    # Use city images from local assets if available
    try:
        st.image(f"assets/{city.lower()}.jpg", width=200)
    except:
        st.image("https://via.placeholder.com/200x150?text=City+Image", width=200)

# Create sample data for map visualization
if city == "Pune":
    sample_data = pd.DataFrame({
        'lat': [18.5204, 18.5304, 18.5404, 18.5504, 18.5604],
        'lon': [73.8567, 73.8667, 73.8767, 73.8867, 73.8967],
        'neighborhood': ['Koregaon Park', 'Shivaji Nagar', 'Kothrud', 'Aundh', 'Yerwada'],
        'hvi': [0.75, 0.68, 0.62, 0.58, 0.65],
        'lst': [36.5, 35.8, 34.9, 34.2, 35.3],
        'population': [12000, 18000, 15000, 13000, 16000],
        'elderly_ratio': [0.08, 0.12, 0.09, 0.07, 0.10],
        'low_income_ratio': [0.15, 0.25, 0.20, 0.18, 0.22],
        'ac_prevalence': [0.45, 0.35, 0.40, 0.50, 0.38],
    })
else:  # Nashik
    sample_data = pd.DataFrame({
        'lat': [19.9975, 20.0075, 20.0175, 20.0275, 20.0375],
        'lon': [73.7898, 73.7998, 73.8098, 73.8198, 73.8298],
        'neighborhood': ['Panchavati', 'Nashik Road', 'CIDCO', 'Gangapur', 'Satpur'],
        'hvi': [0.62, 0.58, 0.55, 0.60, 0.56],
        'lst': [35.2, 34.8, 34.5, 35.0, 34.7],
        'population': [15000, 12000, 18000, 14000, 13000],
        'elderly_ratio': [0.09, 0.11, 0.08, 0.10, 0.07],
        'low_income_ratio': [0.20, 0.18, 0.22, 0.19, 0.17],
        'ac_prevalence': [0.40, 0.42, 0.38, 0.45, 0.43],
    })

# Render map
st.subheader(f"{city} Heat Map Overview")
map_renderer = MapRenderer()
map_renderer.render_map(
    sample_data, 
    city.lower(), 
    map_type=MapType.HYBRID, 
    include_heat_layer=True,
    render_params={"width": 700, "height": 500}
)

# Temperature trends
st.subheader("Temperature Trends (2023)")

# Generate sample temperature data
from datetime import datetime, timedelta
start_date = datetime(2023, 5, 1)
dates = [start_date + timedelta(days=i) for i in range(30)]

if city == "Pune":
    pune_temps = [32 + i*0.1 + (i%5) for i in range(30)]
    nashik_temps = [30 + i*0.08 + (i%7) for i in range(30)]
else:
    pune_temps = [32 + i*0.1 + (i%5) for i in range(30)]
    nashik_temps = [30 + i*0.08 + (i%7) for i in range(30)]

temp_data = pd.DataFrame({
    'Date': dates,
    'Pune': pune_temps,
    'Nashik': nashik_temps
})

# Create temperature comparison chart
chart_renderer = ChartRenderer()
fig = chart_renderer.create_city_comparison_chart(
    pd.DataFrame({
        'metric': ['Avg Temperature', 'Max Temperature', 'Min Temperature'],
        'pune': [np.mean(pune_temps), np.max(pune_temps), np.min(pune_temps)],
        'nashik': [np.mean(nashik_temps), np.max(nashik_temps), np.min(nashik_temps)]
    })
)

st.plotly_chart(fig, use_container_width=True)

# Key findings
st.markdown("---")
st.subheader("Key Findings")
st.markdown("""
- **Pune** shows higher UHI intensity in commercial zones (Koregaon Park: 6.1°C)
- **Nashik** has more uniform heat distribution due to river influence
- Both cities show 2-3°C higher nighttime temperatures in urban cores
- Mitigation through green infrastructure could reduce temperatures by 3.5°C
""")