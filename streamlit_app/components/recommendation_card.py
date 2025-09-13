
from __future__ import annotations

import sys
import os
from pathlib import Path

# Add the parent directory to the path so we can import config and utils
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

import math
from typing import Any, Dict, Iterable, List, Optional
import numpy as np
import pandas as pd
import streamlit as st

# Now import from the parent directory
from config import COLOR_PALETTE, PRIORITY_THRESHOLDS
from utils import to_float, to_int, safe_html, get_priority_bucket

class RecommendationCard:
    def __init__(self):
        self.priority_colors = {
            "high": COLOR_PALETTE["high"],
            "medium": COLOR_PALETTE["medium"],
            "low": COLOR_PALETTE["low"],
        }
    
    def _fmt_number(self, value: Any, fmt: str, default_str: str = "—") -> str:
        """Format a number with the given format string."""
        try:
            if value is None or (isinstance(value, float) and math.isnan(value)):
                return default_str
            return format(float(value), fmt)
        except Exception:
            return default_str
    
    def _fmt_int(self, value: Any, default_str: str = "—") -> str:
        """Format an integer with thousands separators."""
        try:
            if value is None or (isinstance(value, float) and math.isnan(value)):
                return default_str
            return f"{int(float(value)):,}"
        except Exception:
            return default_str
    
    def _normalize_progress(self, value: Any, max_months: float = 24.0) -> float:
        """Normalize implementation time to a progress value between 0.0 and 1.0."""
        v = to_float(value, default=0.0)
        if max_months <= 0:
            max_months = 24.0
        frac = v / max_months
        if not math.isfinite(frac):
            frac = 0.0
        return max(0.0, min(1.0, frac))
    
    def render_recommendation_card(self, recommendation: Dict[str, Any]) -> None:
        """Render a single recommendation card."""
        recommendation = recommendation or {}
        
        # Priority
        priority_score = to_float(recommendation.get("priority_score"), default=0.0)
        bucket, priority_text = get_priority_bucket(priority_score)
        priority_color = self.priority_colors.get(bucket, COLOR_PALETTE["low"])
        
        neighborhood_raw = recommendation.get("neighborhood", "Neighborhood")
        neighborhood = safe_html(str(neighborhood_raw).replace("_", " ").title())
        
        # Header card
        st.markdown(
            f"""
            <div style="
                background-color: {priority_color}20;
                padding: 1rem;
                border-radius: 0.5rem;
                border-left: 4px solid {priority_color};
                margin-bottom: 1rem;
            ">
                <h3 style="margin: 0; color: {priority_color};">
                    {neighborhood}
                </h3>
                <span style="
                    background-color: {priority_color};
                    color: white;
                    padding: 0.2rem 0.5rem;
                    border-radius: 1rem;
                    font-size: 0.8rem;
                    font-weight: bold;
                ">
                    {priority_text}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        # Content columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**📊 Key Metrics**")
            hvi_val = recommendation.get("hvi")
            temp_val = recommendation.get("avg_temperature")
            pop_val = recommendation.get("population")
            st.metric("HVI Score", self._fmt_number(hvi_val, ".2f"))
            st.metric("Temperature", f"{self._fmt_number(temp_val, '.1f', default_str='—')}°C")
            st.metric("Population", self._fmt_int(pop_val))
        
        with col2:
            st.markdown("**💰 Financial Impact**")
            cost_val = recommendation.get("estimated_cost")
            roi_val = recommendation.get("roi")
            pp_val = recommendation.get("payback_period")
            cost_str = self._fmt_int(cost_val)
            st.metric("Cost", f"${cost_str}")
            st.metric("ROI", f"{self._fmt_number(roi_val, '.1f', default_str='—')}%")
            st.metric("Payback Period", f"{self._fmt_number(pp_val, '.1f', default_str='—')} years")
        
        with col3:
            st.markdown("**🎯 Recommended Actions**")
            actions = recommendation.get("recommended_actions") or []
            if isinstance(actions, (list, tuple, pd.Series, np.ndarray)):
                for action in actions:
                    action_name = safe_html(str(action).replace("_", " ").title())
                    st.markdown(f"- {action_name}")
            else:
                st.caption("No actions specified")
        
        # Implementation timeline as progress
        st.markdown("**⏱️ Implementation**")
        timeline_months = recommendation.get("implementation_time", 0)
        progress_frac = self._normalize_progress(timeline_months, max_months=24.0)
        st.progress(progress_frac)
        st.caption(f"Implementation time: {self._fmt_number(timeline_months, '.0f', default_str='0')} months")
        
        # Co-benefits
        st.markdown("**🌟 Co-Benefits**")
        implementation_plan = recommendation.get("implementation_plan") or {}
        actions_detail = implementation_plan.get("actions") if isinstance(implementation_plan, dict) else None
        co_benefits: set[str] = set()
        
        if isinstance(actions_detail, Iterable):
            for action_detail in actions_detail:
                if isinstance(action_detail, dict):
                    action_key = action_detail.get("action")
                else:
                    action_key = action_detail
                action_info = self._get_action_info(action_key)
                for cb in action_info.get("co_benefits", []):
                    co_benefits.add(str(cb))
        
        if co_benefits:
            for benefit in sorted(co_benefits):
                st.markdown(f"- {safe_html(benefit.replace('_', ' ').title())}")
        else:
            st.caption("No co-benefits listed")
    
    def render_summary_cards(self, summary_data: Dict[str, Any]) -> None:
        """Render summary statistics cards."""
        summary_data = summary_data or {}
        total_neighborhoods = to_int(summary_data.get("total_neighborhoods"), default=0)
        total_cost = to_float(summary_data.get("total_cost"), default=0.0)
        total_impact = to_float(summary_data.get("total_impact"), default=0.0)
        total_population = to_float(summary_data.get("total_population"), default=0.0)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(
                f"""
                <div style="background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; text-align: center;">
                    <h3 style="margin: 0; color: {COLOR_PALETTE['primary']};">🏘️</h3>
                    <div style="font-size: 1.5rem; font-weight: bold;">
                        {total_neighborhoods}
                    </div>
                    <div style="font-size: 0.9rem; color: #666;">
                        Total Neighborhoods
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        
        with col2:
            st.markdown(
                f"""
                <div style="background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; text-align: center;">
                    <h3 style="margin: 0; color: {COLOR_PALETTE['secondary']};">💰</h3>
                    <div style="font-size: 1.5rem; font-weight: bold;">
                        ${int(total_cost):,}
                    </div>
                    <div style="font-size: 0.9rem; color: #666;">
                        Total Investment
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        
        with col3:
            st.markdown(
                f"""
                <div style="background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; text-align: center;">
                    <h3 style="margin: 0; color: {COLOR_PALETTE['success']};">🌡️</h3>
                    <div style="font-size: 1.5rem; font-weight: bold;">
                        {total_impact:.1f}°C
                    </div>
                    <div style="font-size: 0.9rem; color: #666;">
                        Avg Temperature Reduction
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        
        with col4:
            st.markdown(
                f"""
                <div style="background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; text-align: center;">
                    <h3 style="margin: 0; color: {COLOR_PALETTE['danger']};">👥</h3>
                    <div style="font-size: 1.5rem; font-weight: bold;">
                        {int(total_population):,}
                    </div>
                    <div style="font-size: 0.9rem; color: #666;">
                        Population Impacted
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    
    def _get_action_info(self, action: Any) -> Dict[str, Any]:
        """Get action information from mitigation actions (placeholder)."""
        # Placeholder; integrate with mitigation engine as needed.
        return {"co_benefits": ["energy_savings", "improved_comfort"]}