
from __future__ import annotations

import sys
import os
from pathlib import Path

# Add the parent directory to the path so we can import config and utils
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Now import from the parent directory
from config import COLOR_PALETTE
from utils import (
    safe_series, non_na_numeric, safe_text,
    is_numeric_series
)

class ChartRenderer:
    def __init__(self):
        self.color_palette = COLOR_PALETTE
    
    def create_vulnerability_dashboard(self, data: pd.DataFrame, city_name: str) -> go.Figure:
        """Create vulnerability factors analysis dashboard with optional LST subplot."""
        data = data.copy() if isinstance(data, pd.DataFrame) else pd.DataFrame()
        
        # Expected columns
        col_pop = safe_series(data, "population")
        col_hvi = safe_series(data, "hvi")
        col_elderly = safe_series(data, "elderly_ratio")
        col_low_income = safe_series(data, "low_income_ratio")
        col_ac = safe_series(data, "ac_prevalence")
        col_lst = safe_series(data, "lst")
        col_nei = safe_series(data, "neighborhood", default=None)
        
        # Determine LST availability
        lst_numeric = non_na_numeric(col_lst)
        has_lst = ("lst" in data.columns) and (len(lst_numeric) > 0)
        
        subplot_titles = (
            "Population vs HVI",
            "Elderly Population vs HVI",
            "Low Income vs HVI",
            "LST vs HVI" if has_lst else "AC Prevalence vs HVI",
        )
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=subplot_titles,
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]],
        )
        
        # Common scatter builder
        def add_scatter(x, y, name, row, col, color_by, colorbar_title):
            x_num = pd.to_numeric(x, errors="coerce")
            y_num = pd.to_numeric(y, errors="coerce")
            c_num = pd.to_numeric(color_by, errors="coerce")
            mask = x_num.notna() & y_num.notna()
            x_num = x_num[mask]
            y_num = y_num[mask]
            c_num = c_num[mask]
            texts = safe_text(col_nei.reindex(x_num.index), fallback_prefix=name)
            
            fig.add_trace(
                go.Scatter(
                    x=x_num, y=y_num,
                    mode="markers",
                    text=texts,
                    marker=dict(
                        color=c_num,
                        colorscale="RdYlGn_r",
                        showscale=True,
                        colorbar=dict(title=colorbar_title),
                        size=10,
                    ),
                    name=name,
                ),
                row=row, col=col,
            )
        
        # Add scatter plots
        add_scatter(col_pop, col_hvi, "Population", 1, 1, color_by=col_hvi, colorbar_title="HVI")
        add_scatter(col_elderly, col_hvi, "Elderly Ratio", 1, 2, color_by=col_hvi, colorbar_title="HVI")
        add_scatter(col_low_income, col_hvi, "Low Income Ratio", 2, 1, color_by=col_hvi, colorbar_title="HVI")
        
        if has_lst:
            add_scatter(col_lst, col_hvi, "LST vs HVI", 2, 2, color_by=col_lst, colorbar_title="LST (°C)")
            x4_label = "LST (°C)"
        else:
            add_scatter(col_ac, col_hvi, "AC Prevalence", 2, 2, color_by=col_hvi, colorbar_title="HVI")
            x4_label = "AC Prevalence"
        
        # Layout and axes
        city_title = str(city_name).title() if isinstance(city_name, str) else "City"
        fig.update_layout(
            title_text=f"Vulnerability Factors Analysis - {city_title}",
            showlegend=False,
            height=800,
            margin=dict(l=40, r=20, t=60, b=40),
        )
        fig.update_xaxes(title_text="Population", row=1, col=1)
        fig.update_xaxes(title_text="Elderly Ratio", row=1, col=2)
        fig.update_xaxes(title_text="Low Income Ratio", row=2, col=1)
        fig.update_xaxes(title_text=x4_label, row=2, col=2)
        
        for r in [1, 2]:
            for c in [1, 2]:
                fig.update_yaxes(title_text="HVI", row=r, col=c)
        
        return fig
    
    def create_city_comparison_chart(self, comparison_data: pd.DataFrame) -> go.Figure:
        """Create city comparison charts with safer indexing and explicit distributions."""
        if not isinstance(comparison_data, pd.DataFrame) or comparison_data.empty:
            empty_fig = go.Figure()
            empty_fig.add_annotation(
                text="No comparison data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
            )
            return empty_fig
        
        # Expected columns
        metric = safe_series(comparison_data, "metric")
        pune = safe_series(comparison_data, "pune")
        nashik = safe_series(comparison_data, "nashik")
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("HVI Comparison", "Temperature Trends", 
                           "Vulnerability Distribution", "Population Impact"),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]],
        )
        
        # HVI/metric comparison as grouped bars
        fig.add_trace(
            go.Bar(x=metric.astype(str), y=pd.to_numeric(pune, errors="coerce"), 
                  name="Pune", marker_color=self.color_palette["primary"]),
            row=1, col=1,
        )
        fig.add_trace(
            go.Bar(x=metric.astype(str), y=pd.to_numeric(nashik, errors="coerce"), 
                  name="Nashik", marker_color=self.color_palette["secondary"]),
            row=1, col=1,
        )
        
        # Temperature trends if LST metrics exist
        metric_str = metric.astype(str).fillna("")
        lst_mask = metric_str.str.contains("LST", case=False, na=False)
        
        if lst_mask.any():
            lst_metrics = metric_str[lst_mask]
            pune_lst = pd.to_numeric(pune[lst_mask], errors="coerce")
            nashik_lst = pd.to_numeric(nashik[lst_mask], errors="coerce")
            
            fig.add_trace(
                go.Scatter(
                    x=lst_metrics, y=pune_lst,
                    name="Pune LST", mode="lines+markers",
                    marker_color=self.color_palette["primary"],
                ),
                row=1, col=2,
            )
            fig.add_trace(
                go.Scatter(
                    x=lst_metrics, y=nashik_lst,
                    name="Nashik LST", mode="lines+markers",
                    marker_color=self.color_palette["secondary"],
                ),
                row=1, col=2,
            )
        
        # Vulnerability distribution
        pune_v = comparison_data.get("pune_vuln_dist", None)
        nashik_v = comparison_data.get("nashik_vuln_dist", None)
        
        def _as_numeric_array(x):
            if isinstance(x, (pd.Series, pd.DataFrame)):
                vals = pd.to_numeric(x.squeeze(), errors="coerce")
                return vals.replace([np.inf, -np.inf], np.nan).dropna().values
            if isinstance(x, (list, tuple, np.ndarray, pd.Index)):
                vals = pd.to_numeric(pd.Series(x), errors="coerce")
                return vals.replace([np.inf, -np.inf], np.nan).dropna().values
            vals = pd.to_numeric(pd.Series(x), errors="coerce")
            return vals.replace([np.inf, -np.inf], np.nan).dropna().values
        
        if pune_v is None or nashik_v is None:
            pune_v = _as_numeric_array(pune)
            nashik_v = _as_numeric_array(nashik)
        else:
            pune_v = _as_numeric_array(pune_v)
            nashik_v = _as_numeric_array(nashik_v)
        
        if len(pune_v) > 0:
            fig.add_trace(
                go.Histogram(
                    x=pune_v, name="Pune",
                    opacity=0.7, nbinsx=20,
                    marker_color=self.color_palette["primary"],
                ),
                row=2, col=1,
            )
        
        if len(nashik_v) > 0:
            fig.add_trace(
                go.Histogram(
                    x=nashik_v, name="Nashik",
                    opacity=0.7, nbinsx=20,
                    marker_color=self.color_palette["secondary"],
                ),
                row=2, col=1,
            )
        
        # Population impact bars
        def _value_for(metric_name: str, series: pd.Series, default=np.nan):
            idx = metric_str[metric_str.str.contains(metric_name, case=False, na=False)].index
            if len(idx) > 0:
                return pd.to_numeric(series.loc[idx], errors="coerce")
            return default
        
        total_pune = _value_for("Total Population", pune)
        highrisk_pune = _value_for("High Risk", pune)
        total_nashik = _value_for("Total Population", nashik)
        highrisk_nashik = _value_for("High Risk", nashik)
        
        def _fallback_pair(series: pd.Series):
            s = pd.to_numeric(series, errors="coerce")
            s = s.replace([np.inf, -np.inf], np.nan)
            total = s.iloc[0] if len(s) > 0 else np.nan
            high = s.iloc[3] if len(s) > 3 else (s.max() if len(s) > 0 else np.nan)
            return total, high
        
        if pd.isna(total_pune) or pd.isna(highrisk_pune):
            total_pune, highrisk_pune = _fallback_pair(pune)
        if pd.isna(total_nashik) or pd.isna(highrisk_nashik):
            total_nashik, highrisk_nashik = _fallback_pair(nashik)
        
        fig.add_trace(
            go.Bar(
                x=["High Risk Population", "Total Population"],
                y=[highrisk_pune, total_pune],
                name="Pune", marker_color=self.color_palette["primary"],
            ),
            row=2, col=2,
        )
        fig.add_trace(
            go.Bar(
                x=["High Risk Population", "Total Population"],
                y=[highrisk_nashik, total_nashik],
                name="Nashik", marker_color=self.color_palette["secondary"],
            ),
            row=2, col=2,
        )
        
        fig.update_layout(
            title_text="Pune vs Nashik: Comprehensive Comparison",
            showlegend=True,
            height=800,
            barmode="group",
            margin=dict(l=40, r=20, t=60, b=40),
        )
        
        return fig
    
    def create_recommendation_chart(self, recommendations: pd.DataFrame) -> go.Figure:
        """Create recommendation impact vs cost chart with robust numeric handling."""
        if not isinstance(recommendations, pd.DataFrame) or recommendations.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="No recommendations available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
            )
            return fig
        
        df = recommendations.copy()
        x = pd.to_numeric(df.get("estimated_cost", pd.Series([])), errors="coerce")
        y = pd.to_numeric(df.get("estimated_impact", pd.Series([])), errors="coerce")
        pscore = pd.to_numeric(df.get("priority_score", pd.Series([])), errors="coerce")
        texts = safe_text(df.get("neighborhood", pd.Series([None] * len(df))), fallback_prefix="Recommendation")
        
        mask = x.notna() & y.notna() & pscore.notna()
        x = x[mask].replace([np.inf, -np.inf], np.nan).dropna()
        y = y[mask].replace([np.inf, -np.inf], np.nan).dropna()
        pscore = pscore[mask].replace([np.inf, -np.inf], np.nan).dropna()
        texts = texts.reindex(x.index).fillna("Recommendation")
        
        if len(x) == 0:
            fig = go.Figure()
            fig.add_annotation(
                text="No valid numeric data to plot",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
            )
            return fig
        
        # Stabilize bubble sizes
        desired_max_px = 40.0
        max_pscore = float(pscore.max()) if len(pscore) else 1.0
        sizeref = 2.0 * max_pscore / (desired_max_px ** 2) if max_pscore > 0 else 0.1
        
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=x, y=y,
                mode="markers",
                text=texts,
                marker=dict(
                    size=pscore,
                    sizemode="area",
                    sizeref=sizeref,
                    sizemin=6,
                    color=pscore,
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(title="Priority Score"),
                    line=dict(width=0.5, color="#333"),
                ),
                name="Recommendations",
            )
        )
        
        # Trend line (only if >= 2 finite points)
        if len(x) > 1 and len(y) > 1:
            x_vals = x.values.astype(float)
            y_vals = y.values.astype(float)
            if np.isfinite(x_vals).all() and np.isfinite(y_vals).all():
                z = np.polyfit(x_vals, y_vals, 1)
                p = np.poly1d(z)
                x_trend = np.linspace(np.min(x_vals), np.max(x_vals), 100)
                y_trend = p(x_trend)
                fig.add_trace(
                    go.Scatter(
                        x=x_trend, y=y_trend,
                        mode="lines",
                        name="Trend Line",
                        line=dict(color=self.color_palette["warning"], dash="dash"),
                    )
                )
        
        fig.update_layout(
            title_text="Recommendations: Impact vs Cost Analysis",
            xaxis_title="Estimated Cost ($)",
            yaxis_title="Estimated Impact (°C)",
            height=600,
            showlegend=True,
            margin=dict(l=40, r=20, t=60, b=40),
        )
        
        return fig