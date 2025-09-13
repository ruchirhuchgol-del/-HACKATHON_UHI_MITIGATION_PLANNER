
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

# Import components directly
from components.map_renderer import MapRenderer, MapType
from components.charts import ChartRenderer
from components.recommendation_card import RecommendationCard

st.set_page_config(
    page_title="Nashik Analysis | UHI Mitigation Planner",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

def app():
    st.title("Nashik Heat Vulnerability Analysis")
    
    # Initialize components
    map_renderer = MapRenderer()
    chart_renderer = ChartRenderer()
    card_renderer = RecommendationCard()
    
    # Create sample data for Nashik
    nashik_data = pd.DataFrame({
        'lat': [19.9975, 20.0075, 20.0175, 20.0275, 20.0375, 20.0475, 20.0575, 20.0675],
        'lon': [73.7898, 73.7998, 73.8098, 73.8198, 73.8298, 73.8398, 73.8498, 73.8598],
        'neighborhood': ['Panchavati', 'Nashik Road', 'CIDCO', 'Gangapur', 'Satpur', 'Deolali', 'Mhasrul', 'Adgaon'],
        'hvi': [0.62, 0.58, 0.55, 0.60, 0.56, 0.52, 0.54, 0.50],
        'sensitivity': [0.55, 0.52, 0.50, 0.54, 0.51, 0.48, 0.49, 0.46],
        'adaptive_capacity': [0.38, 0.40, 0.42, 0.39, 0.41, 0.44, 0.43, 0.45],
        'lst': [35.2, 34.8, 34.5, 35.0, 34.7, 34.3, 34.5, 34.1],
        'population': [15000, 12000, 18000, 14000, 13000, 11000, 10000, 9000],
        'population_over_65': [1800, 1320, 2160, 1540, 1300, 990, 900, 810],
        'low_income_households': [3000, 2400, 3600, 2800, 2600, 2200, 2000, 1800],
        'ac_prevalence': [0.40, 0.42, 0.38, 0.45, 0.43, 0.47, 0.46, 0.50],
        'elderly_ratio': [0.12, 0.11, 0.12, 0.11, 0.10, 0.09, 0.09, 0.09],
        'low_income_ratio': [0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20],
    })
    
    # Calculate summary statistics
    avg_hvi = nashik_data['hvi'].mean()
    max_hvi = nashik_data['hvi'].max()
    high_risk_threshold = 0.7
    high_risk_population = nashik_data[nashik_data['hvi'] >= high_risk_threshold]['population'].sum()
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Neighborhoods", len(nashik_data))
    with col2:
        st.metric("Average HVI", f"{avg_hvi:.2f}")
    with col3:
        st.metric("Maximum HVI", f"{max_hvi:.2f}")
    with col4:
        st.metric("High-Risk Population", f"{high_risk_population:,}")
    
    st.markdown("---")
    
    # Vulnerability map
    st.subheader("Heat Vulnerability Map")
    map_renderer.render_map(
        nashik_data, 
        "nashik", 
        map_type=MapType.VULNERABILITY,
        render_params={"width": 700, "height": 500}
    )
    
    # Vulnerability dashboard
    st.subheader("Vulnerability Factors Analysis")
    dashboard_fig = chart_renderer.create_vulnerability_dashboard(nashik_data, "Nashik")
    st.plotly_chart(dashboard_fig, use_container_width=True)
    
    # Most vulnerable neighborhoods
    st.subheader("Most Vulnerable Neighborhoods")
    most_vulnerable = nashik_data.sort_values('hvi', ascending=False)
    
    # Format for display
    display_data = most_vulnerable.copy()
    display_data['hvi'] = display_data['hvi'].apply(lambda x: f"{x:.2f}")
    display_data['sensitivity'] = display_data['sensitivity'].apply(lambda x: f"{x:.2f}")
    display_data['adaptive_capacity'] = display_data['adaptive_capacity'].apply(lambda x: f"{x:.2f}")
    
    st.dataframe(display_data[['neighborhood', 'hvi', 'sensitivity', 'adaptive_capacity', 'population']])
    
    # Detailed neighborhood analysis
    st.subheader("Neighborhood Details")
    selected_neighborhood = st.selectbox(
        "Select a neighborhood for detailed analysis", 
        nashik_data['neighborhood'].unique()
    )
    
    if selected_neighborhood:
        neighborhood_data = nashik_data[nashik_data['neighborhood'] == selected_neighborhood].iloc[0]
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"### {selected_neighborhood.replace('_', ' ').title()}")
            st.metric("Heat Vulnerability Index", f"{neighborhood_data['hvi']:.2f}")
            st.metric("Sensitivity", f"{neighborhood_data['sensitivity']:.2f}")
            st.metric("Adaptive Capacity", f"{neighborhood_data['adaptive_capacity']:.2f}")
        
        with col2:
            st.markdown("### Demographics")
            st.metric("Population", f"{neighborhood_data['population']:,}")
            st.metric("Population Over 65", f"{neighborhood_data['population_over_65']:,}")
            st.metric("Low Income Households", f"{neighborhood_data['low_income_households']:,}")
            st.metric("AC Prevalence", f"{neighborhood_data['ac_prevalence']*100:.1f}%")
    
    # Mitigation recommendations section
    st.markdown("---")
    st.subheader("Mitigation Recommendations")
    
    # Create sample recommendations
    recommendations = []
    for _, row in most_vulnerable.head(3).iterrows():
        recommendations.append({
            "neighborhood": row['neighborhood'],
            "priority_score": row['hvi'],
            "hvi": row['hvi'],
            "avg_temperature": row['lst'],
            "population": row['population'],
            "estimated_cost": 800000 + (row['hvi'] * 400000),
            "roi": 12.0 + (row['hvi'] * 8),
            "payback_period": 2.5 + (row['hvi'] * 1.5),
            "recommended_actions": ["cool_roofs", "urban_greening", "water_bodies"],
            "implementation_time": 10 + (row['hvi'] * 5),
            "implementation_plan": {
                "actions": [
                    {"action": "cool_roofs", "co_benefits": ["energy_savings", "improved_comfort"]},
                    {"action": "urban_greening", "co_benefits": ["air_quality", "biodiversity"]}
                ]
            }
        })
    
    # Display recommendations
    for rec in recommendations:
        card_renderer.render_recommendation_card(rec)
        st.divider()

if __name__ == "__main__":
    app()