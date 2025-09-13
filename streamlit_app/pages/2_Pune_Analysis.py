
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
import folium
from streamlit_folium import folium_static

# Import components directly
from components.map_renderer import MapRenderer, MapType
from components.charts import ChartRenderer
from components.recommendation_card import RecommendationCard

st.set_page_config(
    page_title="Pune Analysis | UHI Mitigation Planner",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

def app():
    st.title("Pune Heat Vulnerability Analysis")
    
    # Initialize components
    map_renderer = MapRenderer()
    chart_renderer = ChartRenderer()
    card_renderer = RecommendationCard()
    
    # Create sample data for Pune
    pune_data = pd.DataFrame({
        'lat': [18.5204, 18.5304, 18.5404, 18.5504, 18.5604, 18.5704, 18.5804, 18.5904],
        'lon': [73.8567, 73.8667, 73.8767, 73.8867, 73.8967, 73.9067, 73.9167, 73.9267],
        'neighborhood': ['Koregaon Park', 'Shivaji Nagar', 'Kothrud', 'Aundh', 'Yerwada', 'Hadapsar', 'Wakad', 'Baner'],
        'hvi': [0.75, 0.68, 0.62, 0.58, 0.65, 0.70, 0.55, 0.60],
        'sensitivity': [0.65, 0.60, 0.55, 0.50, 0.58, 0.62, 0.48, 0.52],
        'adaptive_capacity': [0.30, 0.35, 0.40, 0.45, 0.38, 0.32, 0.42, 0.40],
        'lst': [36.5, 35.8, 34.9, 34.2, 35.3, 36.2, 34.5, 34.8],
        'population': [12000, 18000, 15000, 13000, 16000, 14000, 12000, 11000],
        'population_over_65': [1200, 2160, 1650, 1170, 1760, 1540, 1080, 990],
        'low_income_households': [1800, 4500, 3750, 2990, 3680, 3920, 2760, 2420],
        'ac_prevalence': [0.45, 0.35, 0.40, 0.50, 0.38, 0.32, 0.42, 0.45],
        'elderly_ratio': [0.10, 0.12, 0.11, 0.09, 0.11, 0.11, 0.09, 0.09],
        'low_income_ratio': [0.15, 0.25, 0.25, 0.23, 0.23, 0.28, 0.23, 0.22],
    })
    
    # Calculate summary statistics
    avg_hvi = pune_data['hvi'].mean()
    max_hvi = pune_data['hvi'].max()
    high_risk_threshold = 0.7
    high_risk_population = pune_data[pune_data['hvi'] >= high_risk_threshold]['population'].sum()
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Neighborhoods", len(pune_data))
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
        pune_data, 
        "pune", 
        map_type=MapType.VULNERABILITY,
        render_params={"width": 700, "height": 500}
    )
    
    # Vulnerability dashboard
    st.subheader("Vulnerability Factors Analysis")
    dashboard_fig = chart_renderer.create_vulnerability_dashboard(pune_data, "Pune")
    st.plotly_chart(dashboard_fig, use_container_width=True)
    
    # Most vulnerable neighborhoods
    st.subheader("Most Vulnerable Neighborhoods")
    most_vulnerable = pune_data.sort_values('hvi', ascending=False)
    
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
        pune_data['neighborhood'].unique()
    )
    
    if selected_neighborhood:
        neighborhood_data = pune_data[pune_data['neighborhood'] == selected_neighborhood].iloc[0]
        
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
            "estimated_cost": 1000000 + (row['hvi'] * 500000),
            "roi": 15.0 + (row['hvi'] * 10),
            "payback_period": 3.0 + (row['hvi'] * 2),
            "recommended_actions": ["cool_roofs", "urban_greening", "water_bodies"],
            "implementation_time": 12 + (row['hvi'] * 6),
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