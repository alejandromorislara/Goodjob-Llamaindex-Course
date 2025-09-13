from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import List, Annotated
from enum import Enum
from datetime import datetime

class SentimentoEnum(str, Enum):
    """Enum para sentimientos posibles"""
    POSITIVO = "positivo"
    NEGATIVO = "negativo" 
    NEUTRAL = "neutral"

class AnalisisFinanciero(BaseModel):
    """
    Modelo para análisis de información financiera extraída de documentos
    """
    model_config = ConfigDict(strict=True, extra='forbid',validate_assignment = True, use_enum_values = True)

    empresa: Annotated[str, Field(min_length=1, max_length=100)] = Field(
        description="Nombre de la empresa analizada"
    )
    trimestre: str = Field(
        description="Trimestre del reporte (ej: Q4 2023)",
        pattern=r"Q[1-4]\s\d{4}"
    )
    ingresos: float = Field(
        description="Ingresos en billones de USD",
        gt=0
    )
    sentimiento: SentimentoEnum = Field(
        description="Sentimiento general del reporte. Debe estar entre positivo, negativo o neutral"
    )
    puntos_clave: List[str] = Field(
        description="Lista de puntos clave extraídos",
        min_items=1,
        max_items=5
    )
    fecha_analisis: str = Field(
        default_factory=lambda: datetime.now().strftime("%Y-%m-%d"),
        description="Fecha del análisis"
    )
    
    @field_validator('ingresos')
    @classmethod
    def validar_ingresos(cls, v):
        """Custom validator for revenue"""
        if v > 1000:  # More than 1 trillion seems unrealistic
            raise ValueError('Los ingresos parecen demasiado altos')
        return v
    
    @field_validator("sentimiento", mode="before")
    @classmethod
    def coerce_sentimiento(cls, v):
        # Acepta strings del LLM y los convierte al Enum
        if isinstance(v, SentimentoEnum):
            return v
        if isinstance(v, str):
            vv = v.strip().lower()
            if vv in {"positivo","negativo","neutral"}:
                return SentimentoEnum(vv)
        raise ValueError("sentimiento debe ser positivo|negativo|neutral")

    @field_validator("puntos_clave", mode="before")
    @classmethod
    def cap_puntos_clave(cls, v):
        # Si el LLM devuelve 6-7 bullets, recorta a 5 antes de validar
        if isinstance(v, list):
            v = [str(x).strip() for x in v if str(x).strip()]
            if not v:
                raise ValueError("puntos_clave no puede estar vacío")
            return v[:5]
        raise ValueError("puntos_clave debe ser una lista")
