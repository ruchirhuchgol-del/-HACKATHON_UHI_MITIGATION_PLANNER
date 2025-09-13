
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Union
import pandas as pd
import asyncio
import io
from datetime import datetime
import json

from src.mitigation_improved import MitigationEngine, PolicyBriefGenerator, PolicyBriefConfig, BriefType
from src.vulnerability import VulnerabilityAnalyzer

app = FastAPI(
    title="Urban Heat Island Mitigation API",
    description="AI-powered urban heat island mitigation planning and policy generation",
    version="2.0.0"
)

# Pydantic models for API
class RecommendationRequest(BaseModel):
    city_name: str = Field(..., description="Name of the city")
    budget_limit: Optional[float] = Field(None, description="Maximum budget constraint")
    priority_threshold: float = Field(0.5, ge=0.0, le=1.0, description="Minimum priority score threshold")
    include_hotspot_analysis: bool = Field(True, description="Include hotspot analysis")
    
    @validator('city_name')
    def validate_city_name(cls, v):
        if len(v.strip()) < 2:
            raise ValueError('City name must be at least 2 characters')
        return v.strip().lower()

class PolicyBriefRequest(BaseModel):
    city_name: str = Field(..., description="Name of the city")
    max_recommendations: int = Field(5, ge=1, le=10, description="Maximum number of recommendations")
    brief_type: str = Field("llm", description="Brief generation type: llm, template, or hybrid")
    include_charts: bool = Field(False, description="Include data visualizations")
    custom_prompt: Optional[str] = Field(None, description="Custom prompt for LLM generation")
    
    @validator('brief_type')
    def validate_brief_type(cls, v):
        if v not in ['llm', 'template', 'hybrid']:
            raise ValueError('Brief type must be: llm, template, or hybrid')
        return v

class RecommendationResponse(BaseModel):
    neighborhood: str
    priority_score: float
    vulnerability_score: float
    lst_temperature: float
    population: int
    recommended_actions: List[str]
    estimated_cost: float
    estimated_impact: float
    roi: float
    implementation_time: int
    co_benefits: List[str]

class MitigationAnalysisResponse(BaseModel):
    city_name: str