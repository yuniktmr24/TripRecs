from pydantic_settings import BaseSettings
from typing import Optional, List, Dict, Any
from pydantic.class_validators import validator

class Config(BaseSettings):
    # this is the path prefix
    API_STR: str = "/triprecs-backend"

    # CORS_ORIGINS: str
    # CORS_ORIGINS_LIST: Optional[List[str]] = None

    """ # This will build the CORS_ORIGINS list
    @validator("CORS_ORIGINS_LIST", pre=True)
    def build_cors_origins(
        cls, v: Optional[List], values: Dict[str, Any]
    ) -> Any:
        if isinstance(v, List):
            return v
        cors_origins = values.get("CORS_ORIGINS")
        if not cors_origins:
            return []
        else:
            cors_list = [origin.strip() for origin in cors_origins.split(",")]
            return cors_list """
        
    class Config:
        case_sensitive = True


config = Config()