
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
from datetime import datetime

# Import components directly
from components.map_renderer import MapRenderer, MapType
from components.charts import ChartRenderer
from components.recommendation_card import RecommendationCard

st.set_page_config(
    page_title="Mitigation Planner | UHI Mitigation Planner",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Placeholder classes for components that would be imported from src
class VulnerabilityAnalyzer:
    def get_combined_data(self, city):
        if city == "pune":
            return pd.DataFrame({
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
        else:  # nashik
            return pd.DataFrame({
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
    
    def get_vulnerability_summary(self, city):
        data = self.get_combined_data(city)
        return {
            'avg_hvi': data['hvi'].mean(),
            'max_hvi': data['hvi'].max(),
            'high_risk_population': data[data['hvi'] >= 0.7]['population'].sum()
        }

class HotspotDetector:
    def create_comprehensive_analysis(self, city):
        # Simulate hotspot analysis
        return {
            'summary': {
                'observation_count': 100,
                'max_observed_temp': 42.5,
                'avg_max_temp': 38.2,
            },
            'persistent_hotspots': [
                {'lat': 18.5304, 'lon': 73.8667, 'intensity': 0.8},
                {'lat': 18.5704, 'lon': 73.9067, 'intensity': 0.75},
            ]
        }

class MitigationEngine:
    def __init__(self):
        self.mitigation_actions = {
            'cool_roofs': {'impact': 2.5, 'cost_per_sqm': 15},
            'urban_greening': {'impact': 3.0, 'cost_per_sqm': 25},
            'water_bodies': {'impact': 2.0, 'cost_per_sqm': 40},
            'permeable_pavements': {'impact': 1.5, 'cost_per_sqm': 30},
            'shade_structures': {'impact': 2.2, 'cost_per_sqm': 20}
        }
    
    def get_recommendations(self, vulnerability_data, hotspot_analysis, budget):
        recommendations = []
        
        # Sort neighborhoods by vulnerability score
        sorted_data = vulnerability_data.sort_values('hvi', ascending=False)
        
        # Calculate hotspot proximity for each neighborhood
        hotspots = hotspot_analysis.get('persistent_hotspots', [])
        
        for _, row in sorted_data.head(5).iterrows():
            # Calculate hotspot proximity (simplified)
            min_distance = float('inf')
            for hotspot in hotspots:
                distance = np.sqrt((row['lat'] - hotspot['lat'])**2 + (row['lon'] - hotspot['lon'])**2)
                min_distance = min(min_distance, distance)
            
            hotspot_proximity = 1.0 / (1.0 + min_distance * 100)  # Convert to proximity score
            
            # Calculate priority score
            priority_score = (row['hvi'] * 0.6) + (hotspot_proximity * 0.4)
            
            # Select appropriate actions based on vulnerability profile
            if row['sensitivity'] > 0.6:
                actions = ['cool_roofs', 'shade_structures']
            elif row['adaptive_capacity'] < 0.4:
                actions = ['urban_greening', 'water_bodies']
            else:
                actions = ['cool_roofs', 'urban_greening']
            
            # Calculate estimated impact and cost
            total_impact = sum(self.mitigation_actions[action]['impact'] for action in actions)
            total_cost = sum(self.mitigation_actions[action]['cost_per_sqm'] * 1000 for action in actions)  # Assume 1000 sqm
            
            # Calculate ROI and payback period
            energy_savings = total_impact * 0.1 * row['population'] * 12  # Simplified calculation
            roi = (energy_savings / total_cost) * 100 if total_cost > 0 else 0
            payback_period = total_cost / energy_savings if energy_savings > 0 else 10
            
            recommendations.append({
                'neighborhood': row['neighborhood'],
                'vulnerability_score': row['hvi'],
                'hotspot_proximity': hotspot_proximity,
                'priority_score': priority_score,
                'estimated_impact': total_impact,
                'implementation_time': 6 + len(actions) * 3,  # Months
                'estimated_cost': total_cost,
                'roi': roi,
                'recommended_actions': actions
            })
        
        # Filter recommendations by budget
        affordable_recommendations = []
        remaining_budget = budget
        
        for rec in sorted(recommendations, key=lambda x: x['priority_score'], reverse=True):
            if rec['estimated_cost'] <= remaining_budget:
                affordable_recommendations.append(rec)
                remaining_budget -= rec['estimated_cost']
        
        return affordable_recommendations

class PolicyBriefGenerator:
    def generate_policy_brief(self, city, recommendations, vulnerability_summary, hotspot_analysis):
        # Generate a simple policy brief
        brief = f"""
# UHI Mitigation Policy Brief for {city}

## Executive Summary
This policy brief provides recommendations for mitigating urban heat island effects in {city} based on vulnerability analysis and hotspot detection.

## Key Findings
- Average HVI: {vulnerability_summary['avg_hvi']:.2f}
- Maximum HVI: {vulnerability_summary['max_hvi']:.2f}
- High-risk population: {vulnerability_summary['high_risk_population']:,}
- Maximum observed temperature: {hotspot_analysis['summary']['max_observed_temp']:.1f}°C

## Recommended Interventions
"""
        
        for i, rec in enumerate(recommendations, 1):
            brief += f"""
### {i}. {rec['neighborhood'].replace('_', ' ').title()}
- **Priority Score**: {rec['priority_score']:.2f}
- **Estimated Impact**: {rec['estimated_impact']:.1f}°C temperature reduction
- **Implementation Time**: {rec['implementation_time']} months
- **Estimated Cost**: ${rec['estimated_cost']:,.0f}
- **ROI**: {rec['roi']:.1f}%
- **Recommended Actions**: {', '.join([a.replace('_', ' ').title() for a in rec['recommended_actions']])}

"""
        
        brief += """
## Implementation Strategy
1. **Short-term (0-12 months)**: Implement cool roofs and shade structures in high-priority areas
2. **Medium-term (1-3 years)**: Develop urban green spaces and water bodies
3. **Long-term (3-5 years)**: Comprehensive city-wide UHI mitigation program

## Monitoring and Evaluation
Establish a monitoring framework to track temperature changes, health outcomes, and energy savings.

## Conclusion
Implementing these recommendations will significantly reduce UHI effects in {city}, improving quality of life and building climate resilience.

---
*Generated on {datetime.now().strftime('%Y-%m-%d')}*
"""
        return brief

def app():
    st.title("Mitigation Planning & Policy Briefs")
    
    # Initialize components
    vulnerability_analyzer = VulnerabilityAnalyzer()
    hotspot_detector = HotspotDetector()
    mitigation_engine = MitigationEngine()
    policy_generator = PolicyBriefGenerator()
    map_renderer = MapRenderer()
    card_renderer = RecommendationCard()
    
    # City selection
    city = st.selectbox("Select City", ["Pune", "Nashik"])
    
    # Get vulnerability data
    vulnerability_data = vulnerability_analyzer.get_combined_data(city.lower())
    vulnerability_summary = vulnerability_analyzer.get_vulnerability_summary(city.lower())
    
    # Budget input
    budget = st.number_input("Available Budget (USD)", min_value=0, value=1000000, step=100000)
    
    # Process MODIS data and detect hotspots
    if st.button("Analyze Heat Hotspots"):
        with st.spinner("Processing MODIS data and detecting hotspots..."):
            hotspot_analysis = hotspot_detector.create_comprehensive_analysis(city.lower())
            
            # Display hotspot analysis
            st.subheader(f"Heat Hotspot Analysis - {city}")
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Observations", hotspot_analysis['summary']['observation_count'])
            with col2:
                st.metric("Max Temperature", f"{hotspot_analysis['summary']['max_observed_temp']:.1f}°C")
            with col3:
                st.metric("Avg Max Temp", f"{hotspot_analysis['summary']['avg_max_temp']:.1f}°C")
            with col4:
                st.metric("Persistent Hotspots", len(hotspot_analysis.get('persistent_hotspots', [])))
            
            # Display hotspot map
            st.subheader("Heat Hotspot Map")
            hotspot_data = vulnerability_data.copy()
            
            # Add hotspot intensity to the data
            hotspots = hotspot_analysis.get('persistent_hotspots', [])
            for _, row in hotspot_data.iterrows():
                min_distance = float('inf')
                max_intensity = 0
                
                for hotspot in hotspots:
                    distance = np.sqrt((row['lat'] - hotspot['lat'])**2 + (row['lon'] - hotspot['lon'])**2)
                    min_distance = min(min_distance, distance)
                    if distance < 0.01:  # Very close
                        max_intensity = max(max_intensity, hotspot['intensity'])
                
                # Add hotspot intensity as a new column
                hotspot_data.loc[_, 'hotspot_intensity'] = max_intensity
            
            map_renderer.render_map(
                hotspot_data,
                city.lower(),
                map_type=MapType.TEMPERATURE,
                include_heat_layer=True,
                render_params={"width": 700, "height": 500}
            )
            
            # Generate recommendations
            recommendations = mitigation_engine.get_recommendations(
                vulnerability_data, 
                hotspot_analysis,
                budget
            )
            
            # Display recommendations
            st.subheader(f"Mitigation Recommendations for {city}")
            
            # Calculate summary statistics
            total_cost = sum(rec['estimated_cost'] for rec in recommendations)
            total_impact = sum(rec['estimated_impact'] for rec in recommendations)
            total_population = sum(vulnerability_data['population'])
            
            # Display summary cards
            summary_data = {
                "total_neighborhoods": len(recommendations),
                "total_cost": total_cost,
                "total_impact": total_impact,
                "total_population": total_population
            }
            
            card_renderer.render_summary_cards(summary_data)
            
            # Display individual recommendations
            for rec in recommendations:
                # Format recommendation for display
                display_rec = {
                    "neighborhood": rec['neighborhood'],
                    "priority_score": rec['priority_score'],
                    "hvi": rec['vulnerability_score'],
                    "avg_temperature": vulnerability_data[
                        vulnerability_data['neighborhood'] == rec['neighborhood']
                    ]['lst'].values[0],
                    "population": vulnerability_data[
                        vulnerability_data['neighborhood'] == rec['neighborhood']
                    ]['population'].values[0],
                    "estimated_cost": rec['estimated_cost'],
                    "roi": rec['roi'],
                    "payback_period": rec['estimated_cost'] / (rec['estimated_impact'] * 0.1 * 
                        vulnerability_data[
                            vulnerability_data['neighborhood'] == rec['neighborhood']
                        ]['population'].values[0] * 12),
                    "recommended_actions": rec['recommended_actions'],
                    "implementation_time": rec['implementation_time'],
                    "implementation_plan": {
                        "actions": [{"action": action} for action in rec['recommended_actions']]
                    }
                }
                
                card_renderer.render_recommendation_card(display_rec)
                st.divider()
            
            # Generate policy brief
            if st.button("Generate Policy Brief"):
                with st.spinner("Generating comprehensive policy brief..."):
                    policy_brief = policy_generator.generate_policy_brief(
                        city, 
                        recommendations, 
                        vulnerability_summary,
                        hotspot_analysis
                    )
                    
                    st.subheader("Policy Brief")
                    st.text_area("Policy Document", policy_brief, height=600)
                    
                    # Download button
                    st.download_button(
                        label="Download Policy Brief",
                        data=policy_brief,
                        file_name=f"{city}_UHI_Mitigation_Policy_Brief.txt",
                        mime="text/plain"
                    )

if __name__ == "__main__":
    app()