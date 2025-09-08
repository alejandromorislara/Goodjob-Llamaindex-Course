"""
🔧 Template: Validación Avanzada de Datos Sintéticos con Pydantic - Control de LLMs

INSTRUCCIONES:
1. Completa las clases marcadas con # TODO
2. Configura las variables de entorno en .env
3. Ejecuta y prueba los casos de validación

ESTUDIANTE: ___________________
FECHA: _______________________
"""

import os
import sys
import json
import re
import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import List, Dict, Optional, Union, Annotated, Any
from uuid import uuid4

# Pydantic imports
from pydantic import BaseModel, Field, validator, root_validator, EmailStr
from pydantic.types import constr, conint, confloat
from pydantic import ValidationError
from dotenv import load_dotenv

# Agregar src al path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Cargar variables de entorno
load_dotenv()

# LlamaIndex imports
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.core.program import LLMTextCompletionProgram

# =============================================================================
# ENUMS Y TIPOS BASE
# =============================================================================

class CustomerType(str, Enum):
    """TODO: Tipos de cliente con límites específicos"""
    BASIC = "___________"      # TODO: ¿Valor para cliente básico?
    STANDARD = "___________"   # TODO: ¿Valor para cliente estándar?
    PREMIUM = "___________"    # TODO: ¿Valor para cliente premium?

class Currency(str, Enum):
    """TODO: Monedas soportadas con reglas específicas"""
    EUR = "___________"        # TODO: ¿Código para Euro?
    USD = "___________"        # TODO: ¿Código para Dólar?
    JPY = "___________"        # TODO: ¿Código para Yen?
    GBP = "___________"        # TODO: ¿Código para Libra?

class DocumentType(str, Enum):
    """TODO: Tipos de documento permitidos"""
    INVOICE = "___________"    # TODO: ¿Valor para factura?
    CONTRACT = "___________"   # TODO: ¿Valor para contrato?
    REPORT = "___________"     # TODO: ¿Valor para reporte?
    EMAIL = "___________"      # TODO: ¿Valor para email?

class PaymentMethodType(str, Enum):
    """TODO: Tipos de método de pago"""
    CARD = "___________"       # TODO: ¿Valor para tarjeta?
    SEPA = "___________"       # TODO: ¿Valor para SEPA?
    CRYPTO = "___________"     # TODO: ¿Valor para crypto?

class ValidationStatus(str, Enum):
    """TODO: Estados de validación"""
    SUCCESS = "___________"    # TODO: ¿Valor para éxito?
    FAILED = "___________"     # TODO: ¿Valor para fallo?
    RETRY = "___________"      # TODO: ¿Valor para reintento?

# =============================================================================
# CLASE 1: CONFIGURACIÓN DEL SISTEMA (1 punto)
# =============================================================================

class SystemConfig:
    """TODO: Configuración y verificación del entorno Pydantic"""
    
    @staticmethod
    def verify_pydantic_environment() -> bool:
        """TODO: Verificar versión de Pydantic v2 y dependencias"""
        print("🔍 Verificando entorno Pydantic...")
        
        try:
            import pydantic
            version = pydantic.VERSION
            print(f"✅ Pydantic v{version}")
            
            # TODO: Verificar que sea v2
            if not version.startswith("2."):
                print(f"❌ Se requiere Pydantic v2.x, encontrado v{version}")
                return False
            
            # TODO: Verificar dependencias adicionales
            required_packages = [
                "___________", "___________", "___________"  # TODO: ¿Qué paquetes verificar?
            ]
            
            for package in required_packages:
                try:
                    __import__(package.replace("-", "_"))
                    print(f"✅ {package}")
                except ImportError:
                    print(f"❌ {package}")
                    return False
            
            return True
            
        except ImportError:
            print("❌ Pydantic no está instalado")
            return False
    
    @staticmethod
    def setup_llm_for_structured_output():
        """TODO: Configurar LLM para salida estructurada con Pydantic"""
        Settings.llm = OpenAI(
            model="___________",           # TODO: ¿Qué modelo usar?
            temperature=___,              # TODO: ¿Qué temperatura para datos sintéticos?
            max_tokens=___,               # TODO: ¿Cuántos tokens máximo?
            api_key=os.getenv("___________")  # TODO: ¿Variable de entorno?
        )
        print("✅ LLM configurado para salida estructurada")

# =============================================================================
# CLASE 2: MODELO DE CLIENTE AVANZADO (2 puntos)
# =============================================================================

class EnhancedCustomer(BaseModel):
    """TODO: Modelo de cliente con validaciones avanzadas"""
    
    # TODO: Campos básicos con constraints
    id: Annotated[int, Field(ge=___, le=___)] = Field(description="___________")  # TODO: ¿Rango de ID?
    email: EmailStr = Field(description="___________")  # TODO: ¿Descripción?
    name: Annotated[str, Field(min_length=___, max_length=___)] = Field(description="___________")  # TODO: ¿Límites de nombre?
    age: Annotated[int, Field(ge=___, le=___)] = Field(description="___________")  # TODO: ¿Rango de edad?
    customer_type: CustomerType = Field(description="___________")  # TODO: ¿Descripción?
    credit_limit: Decimal = Field(description="___________")  # TODO: ¿Descripción?
    is_active: bool = Field(default=True, description="___________")  # TODO: ¿Descripción?
    referral_code: Optional[str] = Field(default=None, description="___________")  # TODO: ¿Descripción?
    signup_date: datetime = Field(description="___________")  # TODO: ¿Descripción?
    
    class Config:
        # TODO: Configuración del modelo
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            Decimal: lambda v: str(v)
        }
    
    @validator('email')
    def validate_business_email(cls, v):
        """TODO: Validar que el email sea de dominio empresarial"""
        business_domains = ['.com', '.org', '.es', '.net', '.edu']
        if not any(domain in v for domain in business_domains):
            raise ValueError('___________')  # TODO: ¿Mensaje de error?
        return v.lower()
    
    @validator('name')
    def normalize_name(cls, v):
        """TODO: Normalizar nombre a Title Case"""
        return v.strip().title()
    
    @validator('age')
    def validate_active_customer_age(cls, v, values):
        """TODO: Validar edad para clientes activos"""
        if values.get('is_active', True) and (v < ___ or v > ___):  # TODO: ¿Rango para activos?
            raise ValueError('___________')  # TODO: ¿Mensaje de error?
        return v
    
    @validator('credit_limit')
    def validate_credit_by_type(cls, v, values):
        """TODO: Validar límite de crédito según tipo de cliente"""
        customer_type = values.get('customer_type')
        limits = {
            CustomerType.BASIC: ___,      # TODO: ¿Límite básico?
            CustomerType.STANDARD: ___,   # TODO: ¿Límite estándar?
            CustomerType.PREMIUM: ___     # TODO: ¿Límite premium?
        }
        
        if customer_type and v > limits.get(customer_type, 0):
            raise ValueError(f'___________')  # TODO: ¿Mensaje de error?
        return v
    
    @validator('referral_code')
    def validate_referral_format(cls, v):
        """TODO: Validar formato de código de referido"""
        if v is not None:
            # TODO: Formato: REF_XXXXXXXX (REF_ + 8 caracteres alfanuméricos)
            pattern = r'^REF_[A-Z0-9]{8}$'
            if not re.match(pattern, v):
                raise ValueError('___________')  # TODO: ¿Mensaje de error?
        return v
    
    @validator('signup_date')
    def validate_signup_date(cls, v):
        """TODO: Validar que la fecha de registro no sea futura"""
        if v > datetime.now():
            raise ValueError('___________')  # TODO: ¿Mensaje de error?
        return v

# =============================================================================
# CLASE 3: VALIDADOR DE SALIDAS LLM (2.5 puntos)
# =============================================================================

class ValidationResult(BaseModel):
    """TODO: Resultado de validación con métricas"""
    status: ValidationStatus = Field(description="___________")  # TODO: ¿Descripción?
    attempts: int = Field(description="___________")  # TODO: ¿Descripción?
    errors: List[str] = Field(default_factory=list, description="___________")  # TODO: ¿Descripción?
    success_rate: float = Field(description="___________")  # TODO: ¿Descripción?
    processing_time: float = Field(description="___________")  # TODO: ¿Descripción?

class LLMDataValidator:
    """TODO: Validador de salidas LLM con retry automático"""
    
    def __init__(self):
        self.validation_stats = {
            'total_attempts': 0,
            'successful_validations': 0,
            'failed_validations': 0,
            'retry_attempts': 0
        }
    
    def validate_llm_output(self, data: Dict[str, Any], model_class: BaseModel) -> ValidationResult:
        """TODO: Validar salida LLM con manejo de errores"""
        start_time = datetime.now()
        attempts = 0
        errors = []
        
        while attempts < ___:  # TODO: ¿Máximo intentos?
            attempts += 1
            self.validation_stats['total_attempts'] += 1
            
            try:
                # TODO: Intentar validar con el modelo Pydantic
                validated_data = model_class.model_validate(data)
                
                # TODO: Éxito - actualizar estadísticas
                self.validation_stats['successful_validations'] += 1
                processing_time = (datetime.now() - start_time).total_seconds()
                
                return ValidationResult(
                    status=ValidationStatus.SUCCESS,
                    attempts=attempts,
                    errors=[],
                    success_rate=self._calculate_success_rate(),
                    processing_time=processing_time
                )
                
            except ValidationError as e:
                # TODO: Error de validación - registrar y continuar
                error_msg = f"Intento {attempts}: {str(e)}"
                errors.append(error_msg)
                print(f"⚠️ {error_msg}")
                
                if attempts < ___:  # TODO: ¿Máximo intentos?
                    self.validation_stats['retry_attempts'] += 1
                    print(f"🔄 Reintentando validación...")
        
        # TODO: Falló después de todos los intentos
        self.validation_stats['failed_validations'] += 1
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return ValidationResult(
            status=ValidationStatus.FAILED,
            attempts=attempts,
            errors=errors,
            success_rate=self._calculate_success_rate(),
            processing_time=processing_time
        )
    
    def _calculate_success_rate(self) -> float:
        """TODO: Calcular tasa de éxito de validación"""
        total = self.validation_stats['total_attempts']
        if total == 0:
            return 0.0
        return self.validation_stats['successful_validations'] / total

# =============================================================================
# CLASE 4: GENERADOR DE DATOS SINTÉTICOS (2 puntos)
# =============================================================================

class SyntheticDataGenerator:
    """TODO: Generador con control de calidad y estadísticas"""
    
    def __init__(self):
        self.validator = LLMDataValidator()
        self.generation_stats = {
            'customers_generated': 0,
            'transactions_generated': 0,
            'total_generation_time': 0.0
        }
    
    def generate_enhanced_customers(self, count: int = 3) -> tuple:
        """TODO: Generar clientes con validación avanzada"""
        print(f"🤖 Generando {count} clientes con validación avanzada...")
        start_time = datetime.now()
        
        # TODO: Crear programa LLM para clientes
        customer_program = LLMTextCompletionProgram.from_defaults(
            output_cls=EnhancedCustomer,
            prompt_template_str="""
            Genera datos realistas para un cliente empresarial.
            
            Requisitos:
            - ID único entre 1000-9999
            - Email empresarial (.com, .org, .es, .net, .edu)
            - Nombre realista en formato correcto
            - Edad entre 18-65 años para clientes activos
            - Tipo de cliente: basic, standard, o premium
            - Límite de crédito apropiado para el tipo
            - Fecha de registro realista (último año)
            - Código de referido opcional formato REF_XXXXXXXX
            
            Haz que los datos sean diversos y realistas.
            """,
            llm=Settings.llm
        )
        
        customers = []
        validation_results = []
        
        for i in range(count):
            print(f"   Generando cliente {i+1}/{count}...")
            
            try:
                # TODO: Generar cliente
                customer = customer_program()
                
                # TODO: Validar resultado
                validation_result = self.validator.validate_llm_output(
                    customer.model_dump() if hasattr(customer, 'model_dump') else customer.__dict__,
                    EnhancedCustomer
                )
                
                if validation_result.status == ValidationStatus.SUCCESS:
                    customers.append(customer)
                    self.generation_stats['customers_generated'] += 1
                
                validation_results.append(validation_result)
                
            except Exception as e:
                print(f"⚠️ Error generando cliente: {e}")
                validation_results.append(ValidationResult(
                    status=ValidationStatus.FAILED,
                    attempts=1,
                    errors=[str(e)],
                    success_rate=0.0,
                    processing_time=0.0
                ))
        
        generation_time = (datetime.now() - start_time).total_seconds()
        self.generation_stats['total_generation_time'] += generation_time
        
        print(f"✅ Generados {len(customers)}/{count} clientes válidos")
        return customers, validation_results

# =============================================================================
# SISTEMA PRINCIPAL DE PRUEBAS
# =============================================================================

class ValidationTestSystem:
    """TODO: Sistema principal para ejecutar pruebas de validación"""
    
    def __init__(self):
        self.generator = SyntheticDataGenerator()
    
    def run_all_tests(self):
        """TODO: Ejecutar todos los casos de prueba"""
        print("🚀 Sistema de Validación Avanzada con Pydantic")
        print("=" * 60)
        
        # TODO: 1. Verificar entorno
        print("\n1️⃣ Verificando entorno...")
        if not SystemConfig.verify_pydantic_environment():
            return False
        
        SystemConfig.setup_llm_for_structured_output()
        
        # TODO: 2. Test de clientes
        print("\n2️⃣ Test: Generación de clientes empresariales...")
        customers, customer_validations = self.generator.generate_enhanced_customers(3)
        self._print_validation_summary("Clientes", customer_validations)
        
        print(f"\n🎉 ¡Sistema de validación completado!")
        print(f"💡 Pydantic garantizó la calidad de todos los datos generados por LLM")
        
        return True
    
    def _print_validation_summary(self, data_type: str, validations: List[ValidationResult]):
        """TODO: Imprimir resumen de validaciones"""
        successful = sum(1 for v in validations if v.status == ValidationStatus.SUCCESS)
        total = len(validations)
        avg_attempts = sum(v.attempts for v in validations) / max(1, total)
        
        print(f"   📊 {data_type}: {successful}/{total} exitosos")
        print(f"   🔄 Intentos promedio: {avg_attempts:.1f}")

# =============================================================================
# FUNCIÓN PRINCIPAL
# =============================================================================

def main():
    """TODO: Función principal del sistema"""
    try:
        system = ValidationTestSystem()
        system.run_all_tests()
        
    except Exception as e:
        print(f"❌ Error general: {e}")
        print("🔍 Revisa tu configuración y variables de entorno")

if __name__ == "__main__":
    print("🔍 Verificando entorno básico...")
    
    # TODO: Verificación rápida
    try:
        import pydantic
        print(f"✅ Pydantic v{pydantic.VERSION}")
    except ImportError:
        print("❌ Pydantic no instalado - Ejecuta: pip install 'pydantic>=2.7'")
        exit(1)
    
    print("\n🚀 Iniciando sistema de validación...")
    main()

# =============================================================================
# PISTAS PARA COMPLETAR EL TEMPLATE
# =============================================================================
"""
🔧 CONFIGURACIÓN:
- Modelo LLM: "gpt-3.5-turbo"
- Temperatura: 0.7 para diversidad
- Max tokens: 1000
- Variable OpenAI: "OPENAI_API_KEY"

📊 RANGOS Y LÍMITES:
- ID cliente: 1000-9999
- Edad activos: 18-65 años
- Crédito Basic: €500, Standard: €2000, Premium: €5000
- Confidence mínimo: 0.7
- Días máximo transacciones: 30
- Límites moneda: EUR/USD €10000, JPY ¥1000000

🔄 ENUMS:
- CustomerType: BASIC="basic", STANDARD="standard", PREMIUM="premium"
- Currency: EUR="EUR", USD="USD", JPY="JPY", GBP="GBP"
- DocumentType: INVOICE="invoice", CONTRACT="contract", REPORT="report", EMAIL="email"
- PaymentMethodType: CARD="card", SEPA="sepa", CRYPTO="crypto"
- ValidationStatus: SUCCESS="success", FAILED="failed", RETRY="retry"

📦 PAQUETES REQUERIDOS:
- "email_validator", "python_dateutil", "requests"

🔢 VALORES NUMÉRICOS:
- Máximo intentos validación: 3
- Confidence threshold alto: 0.85
- Ratio diversidad mínimo: 0.7
- Nombre min/max: 2/50 caracteres
- Título documento: 5/100 caracteres

¡Completa los TODO para crear un sistema robusto de validación! 🚀
"""