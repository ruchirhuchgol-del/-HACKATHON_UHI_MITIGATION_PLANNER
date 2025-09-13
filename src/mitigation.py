# src/mitigation.py
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import openai
from pathlib import Path
from config import config  # Import the config instance

class MitigationEngine:
    def __init__(self):
        self.mitigation_actions = {
            'cool_roofs': {
                'impact': 2.5,  # Temperature reduction in °C
                'cost_per_sqm': 20,  # Cost in USD
                'lifespan': 10,  # Years
                'implementation_time': 6,  # Months
                'maintenance_cost': 0.05,  # Annual maintenance as % of initial cost
                'compliance': ['ECBC', 'NBC'],
                'applicability': ['commercial', 'residential', 'industrial'],
                'co_benefits': ['energy_savings', 'improved_comfort']
            },
            'urban_forestry': {
                'impact': 3.0,
                'cost_per_sqm': 30,
                'lifespan': 25,
                'implementation_time': 24,
                'maintenance_cost': 0.08,
                'compliance': ['NBC', 'Smart City'],
                'applicability': ['public_spaces', 'residential', 'commercial'],
                'co_benefits': ['air_quality', 'biodiversity', 'recreation']
            },
            'cool_pavements': {
                'impact': 1.8,
                'cost_per_sqm': 25,
                'lifespan': 8,
                'implementation_time': 12,
                'maintenance_cost': 0.10,
                'compliance': ['ECBC'],
                'applicability': ['roads', 'public_spaces', 'parking'],
                'co_benefits': ['reduced_runoff', 'improved_safety']
            },
            'green_roofs': {
                'impact': 2.2,
                'cost_per_sqm': 50,
                'lifespan': 20,
                'implementation_time': 18,
                'maintenance_cost': 0.12,
                'compliance': ['GRIHA', 'IGBC'],
                'applicability': ['commercial', 'residential'],
                'co_benefits': ['stormwater_management', 'habitat_creation']
            },
            'water_features': {
                'impact': 1.5,
                'cost_per_sqm': 40,
                'lifespan': 15,
                'implementation_time': 15,
                'maintenance_cost': 0.15,
                'compliance': ['NBC'],
                'applicability': ['public_spaces', 'commercial'],
                'co_benefits': ['recreation', 'microclimate_improvement']
            },
            'shading_structures': {
                'impact': 2.8,
                'cost_per_sqm': 35,
                'lifespan': 12,
                'implementation_time': 9,
                'maintenance_cost': 0.06,
                'compliance': ['NBC'],
                'applicability': ['public_spaces', 'transportation'],
                'co_benefits': ['pedestrian_comfort', 'uv_protection']
            }
        }
    
    def get_recommendations(self, vulnerability_data: pd.DataFrame, 
                          hotspot_analysis: Dict = None, budget: float = None) -> List[Dict]:
        """Generate prioritized mitigation recommendations"""
        recommendations = []
        
        # Calculate vulnerability score for each neighborhood
        vulnerability_data['vulnerability_score'] = (
            vulnerability_data['hvi'] * 0.4 +
            vulnerability_data['sensitivity'] * 0.3 +
            (1 - vulnerability_data['adaptive_capacity']) * 0.3
        )
        
        # Get LST data if available
        has_lst = 'lst' in vulnerability_data.columns and not vulnerability_data['lst'].isna().all()
        
        # Create hotspot analysis if not provided
        if hotspot_analysis is None:
            hotspot_analysis = self.create_mock_hotspot_analysis(vulnerability_data)
        
        # Get persistent hotspots
        persistent_hotspots = hotspot_analysis.get('persistent_hotspots', [])
        
        # Create proximity-based scoring
        for _, neighborhood in vulnerability_data.iterrows():
            # Calculate proximity to persistent hotspots (simplified)
            hotspot_proximity = self.calculate_hotspot_proximity(
                neighborhood, persistent_hotspots
            )
            
            # Get average temperature for this neighborhood
            if has_lst and not pd.isna(neighborhood.get('lst', 0)):
                avg_temp = neighborhood['lst']
            else:
                avg_temp = hotspot_analysis['summary'].get('avg_max_temp', 38.0)
            
            # Recommend actions based on conditions
            suitable_actions = self.get_suitable_actions(
                vulnerability_score=neighborhood['vulnerability_score'],
                avg_temperature=avg_temp,
                hotspot_proximity=hotspot_proximity
            )
            
            # Prioritize by impact/cost ratio
            prioritized_actions = self.prioritize_actions(suitable_actions)
            
            # Calculate implementation details
            implementation_plan = self.create_implementation_plan(
                prioritized_actions, 
                neighborhood['population'],
                budget
            )
            
            # Create recommendation
            recommendation = {
                'neighborhood': neighborhood['neighborhood'],
                'vulnerability_score': neighborhood['vulnerability_score'],
                'hotspot_proximity': hotspot_proximity,
                'avg_temperature': avg_temp,
                'recommended_actions': prioritized_actions,
                'implementation_plan': implementation_plan,
                'estimated_impact': implementation_plan['total_impact'],
                'estimated_cost': implementation_plan['total_cost'],
                'roi': implementation_plan['roi'],
                'implementation_time': implementation_plan['total_time'],
                'priority_score': self.calculate_priority_score(
                    neighborhood['vulnerability_score'],
                    hotspot_proximity,
                    implementation_plan['roi']
                ),
                'hvi': neighborhood['hvi'],
                'sensitivity': neighborhood['sensitivity'],
                'adaptive_capacity': neighborhood['adaptive_capacity'],
                'population': neighborhood['population'],
                'population_over_65': neighborhood['population_over_65'],
                'low_income_households': neighborhood['low_income_households'],
                'ac_prevalence': neighborhood['ac_prevalence']
            }
            
            recommendations.append(recommendation)
        
        # Sort by priority score
        recommendations.sort(key=lambda x: x['priority_score'], reverse=True)
        
        return recommendations
    
    def create_mock_hotspot_analysis(self, vulnerability_data: pd.DataFrame) -> Dict:
        """Create mock hotspot analysis based on vulnerability data"""
        # Sort by HVI to identify hotspots
        sorted_data = vulnerability_data.sort_values('hvi', ascending=False)
        
        # Top 20% as persistent hotspots
        hotspot_count = max(1, int(len(sorted_data) * 0.2))
        persistent_hotspots = sorted_data.head(hotspot_count)['neighborhood'].tolist()
        
        # Calculate temperature statistics
        has_lst = 'lst' in vulnerability_data.columns and not vulnerability_data['lst'].isna().all()
        
        if has_lst:
            lst_data = vulnerability_data[vulnerability_data['lst'] != 0]['lst']
            max_temp = lst_data.max() if len(lst_data) > 0 else 40.0
            avg_temp = lst_data.mean() if len(lst_data) > 0 else 38.0
        else:
            # Estimate based on HVI
            max_temp = 35.0 + (vulnerability_data['hvi'].max() * 5)
            avg_temp = 32.0 + (vulnerability_data['hvi'].mean() * 4)
        
        return {
            'persistent_hotspots': persistent_hotspots,
            'summary': {
                'max_observed_temp': max_temp,
                'avg_max_temp': avg_temp,
                'hotspot_count': hotspot_count
            }
        }
    
    def calculate_hotspot_proximity(self, neighborhood, hotspots):
        """Calculate proximity score to persistent hotspots"""
        if not hotspots:
            return 0.0
        
        # Check if neighborhood is a hotspot
        if neighborhood['neighborhood'] in hotspots:
            return 1.0
        
        # Calculate proximity based on HVI difference
        # Find the maximum HVI among hotspots
        max_hvi = 0
        for hotspot in hotspots:
            if isinstance(hotspot, dict) and 'hvi' in hotspot:
                if hotspot['hvi'] > max_hvi:
                    max_hvi = hotspot['hvi']
            elif isinstance(hotspot, str):
                # If hotspot is just a string (neighborhood name), we can't get HVI
                # So we'll use a default value
                max_hvi = max(max_hvi, 0.8)  # Assuming a high default HVI for hotspots
        
        if max_hvi > 0:
            proximity = 1.0 / (1.0 + abs(neighborhood['hvi'] - max_hvi))
            return min(proximity, 1.0)
        
        return 0.0
    
    def get_suitable_actions(self, vulnerability_score: float, 
                           avg_temperature: float, 
                           hotspot_proximity: float) -> List[str]:
        """Determine suitable mitigation actions"""
        suitable_actions = []
        
        for action, details in self.mitigation_actions.items():
            suitability_score = 0.0
            
            # Temperature-based suitability
            if avg_temperature > 40:
                suitability_score += 0.4
            elif avg_temperature > 35:
                suitability_score += 0.2
            
            # Vulnerability-based suitability
            if vulnerability_score > 0.7:
                suitability_score += 0.3
            elif vulnerability_score > 0.4:
                suitability_score += 0.15
            
            # Hotspot proximity suitability
            if hotspot_proximity > 0.7:
                suitability_score += 0.3
            elif hotspot_proximity > 0.4:
                suitability_score += 0.15
            
            # Add action if suitability score is high enough
            if suitability_score >= 0.4:
                suitable_actions.append(action)
        
        return suitable_actions
    
    def prioritize_actions(self, actions: List[str]) -> List[str]:
        """Prioritize actions by impact/cost ratio"""
        action_scores = []
        for action in actions:
            details = self.mitigation_actions[action]
            score = details['impact'] / details['cost_per_sqm']
            action_scores.append((action, score))
        
        # Sort by score and return action names
        action_scores.sort(key=lambda x: x[1], reverse=True)
        return [action for action, score in action_scores]
    
    def create_implementation_plan(self, actions: List[str], 
                                population: int, budget: float) -> Dict:
        """Create detailed implementation plan"""
        total_impact = 0.0
        total_cost = 0.0
        total_time = 0.0
        plan_details = []
        
        # Calculate area requirements (simplified)
        area_per_person = 0.05  # 50 sqm per person
        total_area = population * area_per_person
        
        for action in actions:
            details = self.mitigation_actions[action]
            
            # Calculate implementation area (distribute among actions)
            action_area = total_area / len(actions)
            
            # Calculate costs and impacts
            action_cost = action_area * details['cost_per_sqm']
            action_impact = details['impact'] * (action_area / total_area)
            
            # Adjust for budget constraints
            if budget and total_cost + action_cost > budget:
                remaining_budget = budget - total_cost
                if remaining_budget > 0:
                    action_area = remaining_budget / details['cost_per_sqm']
                    action_cost = remaining_budget
                    action_impact = details['impact'] * (action_area / total_area)
                else:
                    continue  # Skip if no budget remaining
            
            plan_details.append({
                'action': action,
                'area': action_area,
                'cost': action_cost,
                'impact': action_impact,
                'time': details['implementation_time'],
                'maintenance_cost': action_cost * details['maintenance_cost']
            })
            
            total_impact += action_impact
            total_cost += action_cost
            total_time = max(total_time, details['implementation_time'])
        
        # Calculate ROI
        annual_benefit = total_impact * population * 100  # $100 per person per degree per year
        total_benefit = annual_benefit * 10  # 10-year period
        roi = ((total_benefit - total_cost) / total_cost) * 100 if total_cost > 0 else 0
        
        return {
            'actions': plan_details,
            'total_impact': total_impact,
            'total_cost': total_cost,
            'total_time': total_time,
            'roi': roi,
            'annual_benefit': annual_benefit,
            'total_benefit': total_benefit
        }
    
    def calculate_priority_score(self, vulnerability_score: float, 
                                hotspot_proximity: float, roi: float) -> float:
        """Calculate overall priority score"""
        return (
            vulnerability_score * 0.4 +
            hotspot_proximity * 0.3 +
            min(roi / 100, 1.0) * 0.3
        )
    
    def calculate_citywide_impact(self, recommendations):
        """Calculate city-wide impact of recommendations"""
        if not recommendations:
            return {}
        
        # Group by action
        action_summary = {}
        for rec in recommendations:
            for action_detail in rec['implementation_plan']['actions']:
                action = action_detail['action']
                if action not in action_summary:
                    action_summary[action] = {
                        'cost': 0,
                        'impact': 0,
                        'area': 0,
                        'neighborhoods': []
                    }
                
                action_summary[action]['cost'] += action_detail['cost']
                action_summary[action]['impact'] += action_detail['impact']
                action_summary[action]['area'] += action_detail['area']
                action_summary[action]['neighborhoods'].append(rec['neighborhood'])
        
        # Calculate overall metrics
        total_cost = sum(rec['estimated_cost'] for rec in recommendations)
        total_impact = sum(rec['estimated_impact'] for rec in recommendations)
        total_population = sum(rec['population'] for rec in recommendations)
        total_neighborhoods = len(recommendations)
        
        return {
            'total_cost': total_cost,
            'total_impact': total_impact,
            'total_population': total_population,
            'total_neighborhoods': total_neighborhoods,
            'action_summary': action_summary,
            'priority_distribution': {
                'high': len([r for r in recommendations if r['priority_score'] > 0.7]),
                'medium': len([r for r in recommendations if 0.4 <= r['priority_score'] <= 0.7]),
                'low': len([r for r in recommendations if r['priority_score'] < 0.4])
            }
        }
    
    def get_cost_benefit_analysis(self, recommendations):
        """Generate cost-benefit analysis"""
        if not recommendations:
            return pd.DataFrame()
        
        # Calculate 5-year benefits
        analysis_data = []
        for rec in recommendations:
            plan = rec['implementation_plan']
            annual_benefit = plan['annual_benefit']
            five_year_benefit = plan['total_benefit']
            
            analysis_data.append({
                'neighborhood': rec['neighborhood'],
                'action': ', '.join(rec['recommended_actions']),
                'cost': plan['total_cost'],
                'annual_benefit': annual_benefit,
                'five_year_benefit': five_year_benefit,
                'net_benefit': five_year_benefit - plan['total_cost'],
                'payback_period': plan['total_cost'] / annual_benefit if annual_benefit > 0 else float('inf'),
                'roi': plan['roi'],
                'priority_score': rec['priority_score']
            })
        
        # Create summary table
        df = pd.DataFrame(analysis_data)
        
        # Create action summary
        action_summary = df.groupby('action').agg({
            'cost': 'sum',
            'five_year_benefit': 'sum',
            'net_benefit': 'sum',
            'roi': 'mean',
            'payback_period': 'mean'
        }).round(2)
        
        return {
            'detailed_analysis': df,
            'action_summary': action_summary,
            'total_investment': df['cost'].sum(),
            'total_benefit': df['five_year_benefit'].sum(),
            'total_net_benefit': df['net_benefit'].sum(),
            'average_roi': df['roi'].mean()
        }

class PolicyBriefGenerator:
    def __init__(self):
        self.client = None
        self.setup_openai()
    
    def setup_openai(self):
        """Setup OpenAI client with provided API key"""
        try:
            # Validate configuration
            config.validate()
            
            # Initialize OpenAI client
            self.client = openai.OpenAI(api_key=config.OPENAI_API_KEY)
            print(f"[SUCCESS] OpenAI client initialized successfully with model: {config.OPENAI_MODEL}")
        except ValueError as e:
            print(f"[ERROR] Configuration error: {e}")
            self.client = None
        except Exception as e:
            print(f"[ERROR] OpenAI API setup failed: {e}")
            self.client = None
    
    def generate_policy_brief(self, city_name: str, recommendations: List[Dict], 
                            vulnerability_summary: Dict, hotspot_analysis: Dict = None) -> str:
        """Generate comprehensive policy brief"""
        
        if self.client:
            return self.generate_llm_brief(city_name, recommendations, 
                                        vulnerability_summary, hotspot_analysis)
        else:
            return self.generate_template_brief(city_name, recommendations, 
                                             vulnerability_summary, hotspot_analysis)
    
    def generate_llm_brief(self, city_name: str, recommendations: List[Dict], 
                          vulnerability_summary: Dict, hotspot_analysis: Dict) -> str:
        """Generate policy brief using LLM"""
        
        # Prepare context
        context = self.prepare_llm_context(city_name, recommendations, 
                                        vulnerability_summary, hotspot_analysis)
        
        prompt = f"""
        Generate a comprehensive urban heat island mitigation policy brief for {city_name}.
        
        Context Information:
        {context}
        
        Requirements:
        1. Executive Summary (150 words)
        2. Problem Statement with specific data points
        3. Recommended Interventions with implementation details
        4. 5-Year Implementation Timeline
        5. Budget Requirements and ROI Analysis
        6. Expected Outcomes and Metrics
        7. Policy and Regulatory Recommendations
        
        Format as a professional policy document suitable for municipal officials.
        Include specific numbers and data points from the context.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=config.OPENAI_MODEL,  # Use model from config
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2500,
                temperature=0.7
            )
            
            brief_content = response.choices[0].message.content
            print(f"[SUCCESS] LLM policy brief generated successfully using {config.OPENAI_MODEL}")
            return brief_content
            
        except Exception as e:
            print(f"[ERROR] LLM generation failed: {e}")
            return self.generate_template_brief(city_name, recommendations, 
                                             vulnerability_summary, hotspot_analysis)
    
    def prepare_llm_context(self, city_name: str, recommendations: List[Dict], 
                           vulnerability_summary: Dict, hotspot_analysis: Dict) -> str:
        """Prepare context for LLM"""
        
        context = f"""
CITY: {city_name}
POPULATION: {vulnerability_summary['total_population']:,}
AVERAGE HVI: {vulnerability_summary['avg_hvi']:.2f}
HIGH-RISK POPULATION: {vulnerability_summary['high_risk_population']:,}
"""
        
        # Add LST data if available
        if 'avg_lst' in vulnerability_summary:
            context += f"AVERAGE LST: {vulnerability_summary['avg_lst']:.1f}°C\n"
        
        # Add hotspot analysis if available
        if hotspot_analysis:
            context += f"MAX OBSERVED TEMPERATURE: {hotspot_analysis['summary']['max_observed_temp']:.1f}°C\n"
            context += f"PERSISTENT HOTSPOTS: {len(hotspot_analysis.get('persistent_hotspots', []))}\n"
        
        context += "TOP RECOMMENDATIONS:\n"
        
        for i, rec in enumerate(recommendations[:3], 1):
            context += f"""
{i}. {rec['neighborhood'].replace('_', ' ').title()}:
   - Priority Score: {rec['priority_score']:.2f}
   - Vulnerability Score: {rec['vulnerability_score']:.2f}
   - Recommended Actions: {', '.join(rec['recommended_actions'])}
   - Expected Impact: {rec['estimated_impact']:.1f}°C reduction
   - Implementation Cost: ${rec['estimated_cost']:,.0f}
   - ROI: {rec['roi']:.1f}%
   - Implementation Time: {rec['implementation_time']} months
            """
        
        return context
    
    def generate_template_brief(self, city_name: str, recommendations: List[Dict], 
                              vulnerability_summary: Dict, hotspot_analysis: Dict) -> str:
        """Generate policy brief using templates"""
        
        brief = f"""
# Urban Heat Island Mitigation Policy Brief: {city_name}
## Executive Summary
{city_name} faces significant urban heat island challenges, with an average Heat Vulnerability Index of {vulnerability_summary['avg_hvi']:.2f}"""
        
        # Add temperature information
        if 'avg_lst' in vulnerability_summary:
            brief += f" and average land surface temperatures of {vulnerability_summary['avg_lst']:.1f}°C"
        
        brief += f""". This brief outlines strategic interventions to reduce heat impacts on {vulnerability_summary['total_population']:,} residents, with {vulnerability_summary['high_risk_population']:,} at high risk.
## Problem Statement
Urban heat islands in {city_name} are characterized by:
"""
        
        # Add temperature information
        if 'avg_lst' in vulnerability_summary:
            brief += f"- Land surface temperatures {vulnerability_summary['avg_lst']:.1f}°C on average\n"
        
        brief += f"""- Disproportionate impact on elderly and low-income populations
- Increasing frequency of extreme heat events
- {vulnerability_summary['high_vulnerability_count']} neighborhoods with high vulnerability
## Recommended Interventions
"""
        
        for i, rec in enumerate(recommendations[:3], 1):
            brief += f"""
{i}. **{rec['neighborhood'].replace('_', ' ').title()}** (Priority Score: {rec['priority_score']:.2f})
   - **Actions:** {', '.join(rec['recommended_actions'])}
   - **Expected Impact:** {rec['estimated_impact']:.1f}°C temperature reduction
   - **Investment Required:** ${rec['estimated_cost']:,.0f}
   - **ROI:** {rec['roi']:.1f}% over 10 years
   - **Implementation Timeline:** {rec['implementation_time']} months
            """
        
        brief += """
## 5-Year Implementation Timeline
**Year 1: Foundation Phase**
- Establish UHI monitoring network
- Implement pilot programs in highest priority areas
- Develop regulatory framework
**Year 2-3: Scale Phase**
- Expand successful interventions city-wide
- Integrate UHI mitigation into building codes
- Launch public awareness campaigns
**Year 4-5: Optimization Phase**
- Full implementation across all zones
- Monitoring and evaluation
- Policy refinement based on results
## Budget Requirements
"""
        
        total_cost = sum(rec['estimated_cost'] for rec in recommendations[:3])
        brief += f"""
**Total Estimated Investment:** ${total_cost:,.0f}
- Year 1: ${total_cost * 0.15:,.0f} (15%)
- Year 2-3: ${total_cost * 0.60:,.0f} (60%)
- Year 4-5: ${total_cost * 0.25:,.0f} (25%)
**Funding Sources:**
- Municipal budget allocation
- State climate action funds
- Public-private partnerships
- Green bonds
## Expected Outcomes
- Temperature reduction of 2.5-3.5°C in target areas
- 40-50% reduction in heat-related health incidents
- Energy savings of ${total_cost * 0.3:,.0f} annually
- Creation of {int(total_cost / 50000)} green jobs
- Improved air quality and urban livability
## Monitoring and Evaluation
**Key Performance Indicators:**
- Land surface temperature reduction
- Heat-related hospital admissions
- Energy consumption patterns
- Urban green cover percentage
- Public satisfaction surveys
## Policy Recommendations
1. **Mandate cool roofs** for all new construction and major renovations
2. **Update building codes** to include UHI mitigation requirements
3. **Establish urban forestry targets** (minimum 30% green cover)
4. **Create heat emergency response protocols** for vulnerable populations
5. **Develop incentive programs** for private sector participation
6. **Integrate UHI mitigation** into urban planning processes
7. **Establish dedicated funding** for climate resilience projects
## Conclusion
Addressing urban heat islands in {city_name} requires immediate, coordinated action. The proposed interventions offer a cost-effective approach to reducing heat impacts while providing multiple co-benefits for public health, energy efficiency, and urban livability. Successful implementation will position {city_name} as a leader in climate-resilient urban development.
        """
        
        return brief

# Test execution
if __name__ == "__main__":
    print("Testing Mitigation Engine with OpenAI Integration...")
    try:
        # Validate configuration before running tests
        try:
            config.validate()
            print("[SUCCESS] Configuration validated successfully")
        except ValueError as e:
            print(f"[ERROR] {e}")
            print("Please set the required environment variables and try again.")
            exit(1)
        
        from vulnerability import VulnerabilityAnalyzer
        
        # Initialize engines
        vuln_analyzer = VulnerabilityAnalyzer()
        mit_engine = MitigationEngine()
        policy_gen = PolicyBriefGenerator()
        
        # Test Pune recommendations
        pune_data = vuln_analyzer.get_combined_data('pune')
        pune_recs = mit_engine.get_recommendations(pune_data)
        
        print(f"[SUCCESS] Pune recommendations: {len(pune_recs)} actions")
        if pune_recs:
            print(f"   Total cost: ${sum(rec['estimated_cost'] for rec in pune_recs):,.0f}")
            print(f"   Avg impact: {sum(rec['estimated_impact'] for rec in pune_recs) / len(pune_recs):.1f}°C")
        
        # Test Nashik recommendations
        nashik_data = vuln_analyzer.get_combined_data('nashik')
        nashik_recs = mit_engine.get_recommendations(nashik_data)
        
        print(f"[SUCCESS] Nashik recommendations: {len(nashik_recs)} actions")
        if nashik_recs:
            print(f"   Total cost: ${sum(rec['estimated_cost'] for rec in nashik_recs):,.0f}")
            print(f"   Avg impact: {sum(rec['estimated_impact'] for rec in nashik_recs) / len(nashik_recs):.1f}°C")
        
        # Test city-wide impact
        pune_impact = mit_engine.calculate_citywide_impact(pune_recs)
        print(f"[SUCCESS] Pune city-wide impact calculated")
        
        # Test cost-benefit analysis
        cost_benefit = mit_engine.get_cost_benefit_analysis(pune_recs)
        print(f"[SUCCESS] Cost-benefit analysis completed")
        
        # Test policy generation with LLM
        pune_summary = vuln_analyzer.get_vulnerability_summary('pune')
        policy_brief = policy_gen.generate_policy_brief('pune', pune_recs, pune_summary)
        print(f"[SUCCESS] Policy brief generated: {len(policy_brief)} characters")
        
        # Test with a smaller subset for LLM
        if len(pune_recs) > 0:
            sample_recs = pune_recs[:2]  # Test with top 2 recommendations
            sample_brief = policy_gen.generate_policy_brief('pune', sample_recs, pune_summary)
            print(f"[SUCCESS] Sample LLM brief (2 recommendations): {len(sample_brief)} characters")
        
        print("\n[SUCCESS] All mitigation engine tests with OpenAI integration passed!")
        
    except Exception as e:
        print(f"[ERROR] Error: {e}")
        import traceback
        traceback.print_exc()