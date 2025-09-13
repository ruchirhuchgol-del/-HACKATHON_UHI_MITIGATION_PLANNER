# src/policy_generator.py
class PolicyGenerator:
    def __init__(self):
        self.templates = {
            'executive': "Heat Mitigation Strategy for {city}",
            'technical': "Technical Implementation Plan",
            'community': "Community Engagement Strategy"
        }
    
    def generate_brief(self, city_name, recommendations, template_type='executive'):
        # Simple template-based generation (mock LLM for speed)
        brief = f"""
        {self.templates[template_type]}: {city_name.title()}
        
        Key Recommendations:
        {self._format_recommendations(recommendations)}
        
        Expected Impact: 3.2°C temperature reduction
        Implementation Cost: ₹{sum(r['cost'] for r in recommendations)} crore
        Timeline: 24 months
        """
        return brief