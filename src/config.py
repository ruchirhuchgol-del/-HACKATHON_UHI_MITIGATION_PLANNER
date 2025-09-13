# src/config.py
import os
from typing import Optional

class Config:
    """Configuration class for the UHI Mitigation Planner"""
    
    # OpenAI API configuration
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
    
    # Application settings
    DEBUG_MODE: bool = os.getenv("DEBUG_MODE", "False").lower() == "true"
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    # Data paths
    DATA_DIR: str = os.getenv("DATA_DIR", os.path.join(os.path.dirname(__file__), "..", "data"))
    
    @classmethod
    def validate(cls):
        """Validate required configuration values"""
        missing = []
        if not cls.OPENAI_API_KEY:
            missing.append("OPENAI_API_KEY")
        
        if missing:
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")
        
        return True

# Create a single instance for easy import
config = Config()