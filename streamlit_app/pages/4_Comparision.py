
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
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import components directly
from components.charts import ChartRenderer
from components.map_renderer import MapRenderer, MapType

st.set_page_config(
    page_title="City Comparison | UHI Mitigation Planner",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

def app():
    st.title("Pune vs Nashik: Vulnerability Comparison")
    
    # Initialize components
    chart_renderer = ChartRenderer()
    map_renderer = MapRenderer()
    
    # Create sample data for Pune and Nashik
    pune_data = pd.DataFrame({
        'lat': [18.5204, 18.5304, 18.5404, 18.5504, 18.5604, 18.5704, 18.5804, 18.5904],
        'lon': [73.8567, 73.8667, 73.8767, 73.8867, 73.8967, 73.9067, 73.9167, 73.9267],
        'neighborhood': ['Koregaon Park', 'Shivaji Nagar', 'Kothrud', 'Aundh', 'Yerwada', 'Hadapsar', 'Wakad', 'Baner'],
        'hvi': [0.75, 0.68, 0.62, 0.58, 0.65, 0.70, 0.55, 0.60],
        'sensitivity': [0.65, 0.60, 0.55, 0.50, 0.58, 0.62, 0.48, 0.52],
        'adaptive_capacity': [0.30, 0.35, 0.40, 0.45, 0.38, 0.32, 0.42, 0.40],
        'lst': [36.5, 35.8, 34.9, 34.2, 35.3, 36.2, 34.5, 34.8],
        'population': [12000, 18000, 15000, 13000, 16000, 14000, 12000, 11000],
        'elderly_ratio': [0.10, 0.12, 0.11, 0.09, 0.11, 0.11, 0.09, 0.09],
        'low_income_ratio': [0.15, 0.25, 0.25, 0.23, 0.23, 0.28, 0.23, 0.22],
        'ac_prevalence': [0.45, 0.35, 0.40, 0.50, 0.38, 0.32, 0.42, 0.45],
    })
    
    nashik_data = pd.DataFrame({
        'lat': [19.9975, 20.0075, 20.0175, 20.0275, 20.0375, 20.0475, 20.0575, 20.0675],
        'lon': [73.7898, 73.7998, 73.8098, 73.8198, 73.8298, 73.8398, 73.8498, 73.8598],
        'neighborhood': ['Panchavati', 'Nashik Road', 'CIDCO', 'Gangapur', 'Satpur', 'Deolali', 'Mhasrul', 'Adgaon'],
        'hvi': [0.62, 0.58, 0.55, 0.60, 0.56, 0.52, 0.54, 0.50],
        'sensitivity': [0.55, 0.52, 0.50, 0.54, 0.51, 0.48, 0.49, 0.46],
        'adaptive_capacity': [0.38, 0.40, 0.42, 0.39, 0.41, 0.44, 0.43, 0.45],
        'lst': [35.2, 34.8, 34.5, 35.0, 34.7, 34.3, 34.5, 34.1],
        'population': [15000, 12000, 18000, 14000, 13000, 11000, 10000, 9000],
        'elderly_ratio': [0.12, 0.11, 0.12, 0.11, 0.10, 0.09, 0.09, 0.09],
        'low_income_ratio': [0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20],
        'ac_prevalence': [0.40, 0.42, 0.38, 0.45, 0.43, 0.47, 0.46, 0.50],
    })
    
    # Calculate summary statistics
    pune_avg_hvi = pune_data['hvi'].mean()
    pune_max_hvi = pune_data['hvi'].max()
    pune_high_risk_pop = pune_data[pune_data['hvi'] >= 0.7]['population'].sum()
    pune_avg_elderly = pune_data['elderly_ratio'].mean()
    pune_avg_low_income = pune_data['low_income_ratio'].mean()
    pune_avg_ac = pune_data['ac_prevalence'].mean()
    
    nashik_avg_hvi = nashik_data['hvi'].mean()
    nashik_max_hvi = nashik_data['hvi'].max()
    nashik_high_risk_pop = nashik_data[nashik_data['hvi'] >= 0.7]['population'].sum()
    nashik_avg_elderly = nashik_data['elderly_ratio'].mean()
    nashik_avg_low_income = nashik_data['low_income_ratio'].mean()
    nashik_avg_ac = nashik_data['ac_prevalence'].mean()
    
    # Create comparison dataframe
    comparison_df = pd.DataFrame({
        'metric': [
            'Average HVI', 
            'Maximum HVI', 
            'High Risk Population', 
            'Average Elderly Ratio',
            'Average Low Income Ratio',
            'Average AC Prevalence'
        ],
        'pune': [
            pune_avg_hvi,
            pune_max_hvi,
            pune_high_risk_pop,
            pune_avg_elderly,
            pune_avg_low_income,
            pune_avg_ac
        ],
        'nashik': [
            nashik_avg_hvi,
            nashik_max_hvi,
            nashik_high_risk_pop,
            nashik_avg_elderly,
            nashik_avg_low_income,
            nashik_avg_ac
        ]
    })
    
    # Key metrics comparison
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        diff = pune_avg_hvi - nashik_avg_hvi
        st.metric("Pune Avg HVI", f"{pune_avg_hvi:.2f}", f"{diff:+.2f} vs Nashik")
    
    with col2:
        diff = pune_max_hvi - nashik_max_hvi
        st.metric("Pune Max HVI", f"{pune_max_hvi:.2f}", f"{diff:+.2f} vs Nashik")
    
    with col3:
        diff = pune_high_risk_pop - nashik_high_risk_pop
        st.metric("Pune High-Risk Pop", f"{pune_high_risk_pop:,}", f"{diff:+,} vs Nashik")
    
    with col4:
        diff = pune_avg_elderly - nashik_avg_elderly
        st.metric("Pune Elderly Ratio", f"{pune_avg_elderly:.2f}", f"{diff:+.2f} vs Nashik")
    
    st.markdown("---")
    
    # Comparison table
    st.subheader("Vulnerability Metrics Comparison")
    
    # Format for display
    display_df = comparison_df.copy()
    display_df['pune'] = display_df['pune'].apply(lambda x: f"{x:.2f}" if isinstance(x, float) else f"{x:,}")
    display_df['nashik'] = display_df['nashik'].apply(lambda x: f"{x:.2f}" if isinstance(x, float) else f"{x:,}")
    
    st.dataframe(display_df)
    
    # HVI distribution comparison
    st.subheader("HVI Distribution Comparison")
    
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=pune_data['hvi'],
        name='Pune',
        opacity=0.75,
        nbinsx=20
    ))
    fig.add_trace(go.Histogram(
        x=nashik_data['hvi'],
        name='Nashik',
        opacity=0.75,
        nbinsx=20
    ))
    
    fig.update_layout(
        title_text='HVI Distribution Comparison',
        xaxis_title_text='HVI Value',
        yaxis_title_text='Count',
        barmode='overlay'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Vulnerability factors comparison
    st.subheader("Vulnerability Factors Comparison")
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Elderly Ratio', 'Low Income Ratio', 'AC Prevalence', 'Sensitivity vs Adaptive Capacity'),
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "scatter"}]]
    )
    
    # Elderly ratio
    fig.add_trace(
        go.Bar(x=['Pune', 'Nashik'], y=[pune_avg_elderly, nashik_avg_elderly], name='Elderly Ratio'),
        row=1, col=1
    )
    
    # Low income ratio
    fig.add_trace(
        go.Bar(x=['Pune', 'Nashik'], y=[pune_avg_low_income, nashik_avg_low_income], name='Low Income Ratio'),
        row=1, col=2
    )
    
    # AC prevalence
    fig.add_trace(
        go.Bar(x=['Pune', 'Nashik'], y=[pune_avg_ac, nashik_avg_ac], name='AC Prevalence'),
        row=2, col=1
    )
    
    # Sensitivity vs Adaptive Capacity
    fig.add_trace(
        go.Scatter(
            x=pune_data['adaptive_capacity'], 
            y=pune_data['sensitivity'],
            mode='markers',
            text=pune_data['neighborhood'],
            name='Pune',
            marker=dict(color='red', size=8)
        ),
        row=2, col=2
    )
    
    fig.add_trace(
        go.Scatter(
            x=nashik_data['adaptive_capacity'], 
            y=nashik_data['sensitivity'],
            mode='markers',
            text=nashik_data['neighborhood'],
            name='Nashik',
            marker=dict(color='blue', size=8)
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        title_text="Vulnerability Factors Comparison",
        showlegend=False,
        height=800
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Most vulnerable neighborhoods comparison
    st.subheader("Most Vulnerable Neighborhoods Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Pune")
        pune_most_vulnerable = pune_data.sort_values('hvi', ascending=False).head(3)
        pune_display = pune_most_vulnerable.copy()
        pune_display['hvi'] = pune_display['hvi'].apply(lambda x: f"{x:.2f}")
        st.dataframe(pune_display[['neighborhood', 'hvi', 'sensitivity', 'adaptive_capacity']])
    
    with col2:
        st.markdown("### Nashik")
        nashik_most_vulnerable = nashik_data.sort_values('hvi', ascending=False).head(3)
        nashik_display = nashik_most_vulnerable.copy()
        nashik_display['hvi'] = nashik_display['hvi'].apply(lambda x: f"{x:.2f}")
        st.dataframe(nashik_display[['neighborhood', 'hvi', 'sensitivity', 'adaptive_capacity']])
    
    # City maps comparison
    st.subheader("City Maps Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Pune Vulnerability Map")
        map_renderer.render_map(
            pune_data, 
            "pune", 
            map_type=MapType.VULNERABILITY,
            render_params={"width": 400, "height": 300}
        )
    
    with col2:
        st.markdown("### Nashik Vulnerability Map")
        map_renderer.render_map(
            nashik_data, 
            "nashik", 
            map_type=MapType.VULNERABILITY,
            render_params={"width": 400, "height": 300}
        )

if __name__ == "__main__":
    app()