#!/usr/bin/env python
# coding: utf-8

# # Urban Heat Island Visualization Dashboard
# 
# This notebook provides visualizations for demographic data in Pune and Nashik.

# In[ ]:


# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")


# ## 1. Load and Prepare Data

# In[ ]:


# Define cities and data paths
cities = ['pune', 'nashik']
data_paths = {
    'pune': {
        'demographics': './pune/pune.csv'
    },
    'nashik': {
        'demographics': './nashik/nashik.csv'
    }
}

# Load data for a selected city
selected_city = 'pune'  # Change to 'nashik' for Nashik analysis

# Load demographics
demographics = pd.read_csv(data_paths[selected_city]['demographics'])

print(f"✅ Loaded data for {selected_city.title()}")
print(f"Demographics data shape: {demographics.shape}")
print("Sample data:")
# Line 50: Replace display(demographics.head()) with:
print(demographics.head())


# ## 2. Basic Data Exploration

# In[ ]:


# Display demographics statistics
print(f"Demographics Statistics for {selected_city.title()}:")
print(f"Total neighborhoods: {len(demographics)}")
print(f"Total population: {demographics['population'].sum():,}")

# Convert percentage strings to float values
demographics['elderly_pct'] = demographics['population_over_65'].str.replace('%', '').astype(float)
demographics['low_income_pct'] = demographics['low_income_households'].str.replace('%', '').astype(float)
demographics['ac_pct'] = demographics['ac_prevalence'].str.replace('%', '').astype(float)

print(f"Average elderly population: {demographics['elderly_pct'].mean():.2f}%")
print(f"Average low income households: {demographics['low_income_pct'].mean():.2f}%")
print(f"Average AC prevalence: {demographics['ac_pct'].mean():.2f}%")


# ## 3. Demographic Visualizations

# In[ ]:


# Demographics visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Population distribution
axes[0, 0].bar(demographics['neighborhood'], demographics['population'], color='skyblue')
axes[0, 0].set_title('Population by Neighborhood')
axes[0, 0].set_xticklabels(demographics['neighborhood'], rotation=45, ha='right')

# Elderly population
axes[0, 1].bar(demographics['neighborhood'], demographics['elderly_pct'], color='salmon')
axes[0, 1].set_title('Population Over 65 (%)')
axes[0, 1].set_xticklabels(demographics['neighborhood'], rotation=45, ha='right')

# Low income households
axes[1, 0].bar(demographics['neighborhood'], demographics['low_income_pct'], color='lightgreen')
axes[1, 0].set_title('Low Income Households (%)')
axes[1, 0].set_xticklabels(demographics['neighborhood'], rotation=45, ha='right')

# AC prevalence
axes[1, 1].bar(demographics['neighborhood'], demographics['ac_pct'], color='gold')
axes[1, 1].set_title('AC Prevalence (%)')
axes[1, 1].set_xticklabels(demographics['neighborhood'], rotation=45, ha='right')

plt.tight_layout()
plt.show()


# ## 4. Interactive Plotly Dashboard

# In[ ]:


# Create a comprehensive dashboard
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=('Population Distribution', 'Elderly Population', 
                   'Low Income Households', 'AC Prevalence'),
    specs=[[{"type": "bar"}, {"type": "bar"}],
            [{"type": "bar"}, {"type": "bar"}]]
)

# 1. Population Distribution
fig.add_trace(
    go.Bar(
        x=demographics['neighborhood'],
        y=demographics['population'],
        name='Population',
        marker_color='skyblue'
    ),
    row=1, col=1
)

# 2. Elderly Population
fig.add_trace(
    go.Bar(
        x=demographics['neighborhood'],
        y=demographics['elderly_pct'],
        name='Elderly %',
        marker_color='salmon'
    ),
    row=1, col=2
)

# 3. Low Income Households
fig.add_trace(
    go.Bar(
        x=demographics['neighborhood'],
        y=demographics['low_income_pct'],
        name='Low Income %',
        marker_color='lightgreen'
    ),
    row=2, col=1
)

# 4. AC Prevalence
fig.add_trace(
    go.Bar(
        x=demographics['neighborhood'],
        y=demographics['ac_pct'],
        name='AC Prevalence %',
        marker_color='gold'
    ),
    row=2, col=2
)

# Update layout
fig.update_layout(
    title_text=f"Demographic Analysis - {selected_city.title()}",
    showlegend=True,
    height=800
)

# Update xaxes
fig.update_xaxes(tickangle=45, tickfont=dict(size=10))

# Show the figure
fig.show()

# Save the figure
fig.write_html(f'{selected_city}_dashboard.html')
print(f"✅ Saved interactive dashboard to {selected_city}_dashboard.html")


# ## 5. Heat Vulnerability Index (HVI) Calculation

# In[ ]:


# Calculate HVI components
# Normalize all factors to 0-1 scale
def normalize(series):
    return (series - series.min()) / (series.max() - series.min())

# Sensitivity component (elderly + low income)
sensitivity = normalize(demographics['elderly_pct'] + demographics['low_income_pct'])

# Adaptive capacity component (AC prevalence - inverse)
adaptive_capacity = 1 - normalize(demographics['ac_pct'])

# Calculate HVI (simple weighted average)
weights = {
    'sensitivity': 0.6,
    'adaptive_capacity': 0.4
}

hvi = (weights['sensitivity'] * sensitivity + 
       weights['adaptive_capacity'] * adaptive_capacity)

# Create HVI DataFrame
hvi_df = pd.DataFrame({
    'neighborhood': demographics['neighborhood'],
    'sensitivity': sensitivity,
    'adaptive_capacity': adaptive_capacity,
    'hvi': hvi
})

# Sort by HVI
hvi_df = hvi_df.sort_values('hvi', ascending=False)

print("Heat Vulnerability Index (HVI) Results:")
# Line 218: Replace display(hvi_df) with:
print(hvi_df)

# Visualize HVI
plt.figure(figsize=(12, 6))
bars = plt.bar(hvi_df['neighborhood'], hvi_df['hvi'], color='orangered')
plt.title(f'Heat Vulnerability Index - {selected_city.title()}', fontsize=16)
plt.ylabel('HVI Score (0-1)', fontsize=12)
plt.xticks(rotation=45, ha='right')

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.2f}',
             ha='center', va='bottom')

plt.tight_layout()
plt.show()

# Save HVI results
hvi_df.to_csv(f'{selected_city}_hvi.csv', index=False)
print(f"✅ Saved HVI results to {selected_city}_hvi.csv")


# ## 6. Mitigation Recommendations
# 
# Based on the HVI scores, we can generate prioritized mitigation recommendations.

# In[ ]:


# Define mitigation actions and their effectiveness
mitigation_actions = {
    'tree_planting': {
        'cooling_effect': 2.0,  # °C reduction
        'cost_per_unit': 5000,  # INR per tree
        'feasibility': 0.9,     # 0-1 scale
        'description': 'Plant shade trees in public spaces and along roads'
    },
    'cool_roofs': {
        'cooling_effect': 1.5,
        'cost_per_unit': 200,  # INR per sqm
        'feasibility': 0.8,
        'description': 'Install reflective roofing materials'
    },
    'green_roofs': {
        'cooling_effect': 1.8,
        'cost_per_unit': 1500,  # INR per sqm
        'feasibility': 0.6,
        'description': 'Convert rooftops to vegetated green roofs'
    },
    'cool_pavements': {
        'cooling_effect': 1.2,
        'cost_per_unit': 800,   # INR per sqm
        'feasibility': 0.7,
        'description': 'Use reflective materials for roads and pavements'
    },
    'water_features': {
        'cooling_effect': 1.0,
        'cost_per_unit': 10000, # INR per unit
        'feasibility': 0.5,
        'description': 'Install fountains, ponds, or misting systems'
    }
}

# Generate recommendations for top 3 most vulnerable neighborhoods
top_neighborhoods = hvi_df.head(3)['neighborhood'].values

recommendations = []

for neighborhood in top_neighborhoods:
    # Get neighborhood data
    n_data = hvi_df[hvi_df['neighborhood'] == neighborhood].iloc[0]
    
    # Get demographics
    demo = demographics[demographics['neighborhood'] == neighborhood].iloc[0]
    
    # Calculate priority score for each action
    for action, details in mitigation_actions.items():
        # Priority = cooling_effect * feasibility / cost_per_unit
        priority = (details['cooling_effect'] * details['feasibility']) / details['cost_per_unit']
        
        recommendations.append({
            'neighborhood': neighborhood,
            'hvi_score': n_data['hvi'],
            'population': demo['population'],
            'action': action,
            'description': details['description'],
            'cooling_effect': details['cooling_effect'],
            'cost_per_unit': details['cost_per_unit'],
            'feasibility': details['feasibility'],
            'priority_score': priority
        })

# Create DataFrame and sort by priority
rec_df = pd.DataFrame(recommendations)
rec_df = rec_df.sort_values(['neighborhood', 'priority_score'], ascending=[True, False])

# Display recommendations
print(f"Top Mitigation Recommendations for {selected_city.title()}:")
# Line 318: Replace display(rec_df) with:
print(rec_df)

# Visualize recommendations
plt.figure(figsize=(14, 8))

# Create a grouped bar chart
actions = rec_df['action'].unique()
x = np.arange(len(top_neighborhoods))
width = 0.15

for i, action in enumerate(actions):
    action_data = rec_df[rec_df['action'] == action]
    plt.bar(x + i*width, action_data['priority_score'], width, label=action.replace('_', ' ').title())

plt.title(f'Mitigation Action Priority Scores - {selected_city.title()}', fontsize=16)
plt.ylabel('Priority Score', fontsize=12)
plt.xticks(x + width*2, top_neighborhoods)
plt.legend()
plt.tight_layout()
plt.show()

# Save recommendations
rec_df.to_csv(f'{selected_city}_recommendations.csv', index=False)
print(f"✅ Saved recommendations to {selected_city}_recommendations.csv")

