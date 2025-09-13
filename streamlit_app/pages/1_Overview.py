
from __future__ import annotations

import sys
import os
from pathlib import Path

# Add the parent directory to the path so we can import components
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import components directly
from components.map_renderer import MapRenderer, MapType
from components.charts import ChartRenderer

st.set_page_config(
    page_title="Overview | UHI Mitigation Planner",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

def app():
    st.title("🌡️ Urban Heat Island Mitigation Planner")
    st.markdown("### Climate-Resilient Urban Planning for Indian Cities")
    
    # Initialize components
    map_renderer = MapRenderer()
    chart_renderer = ChartRenderer()
    
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

if __name__ == "__main__":
    app()