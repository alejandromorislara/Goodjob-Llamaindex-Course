# 🔧 Ejercicio: Validación Avanzada de Datos Sintéticos con Pydantic - Control de LLMs

## **Objetivo**
Implementar un sistema avanzado de **validación y generación de datos sintéticos** utilizando **Pydantic** con validaciones personalizadas, constraints avanzados y control estricto de salidas de LLMs para demostrar la importancia del tipado en sistemas de IA.

## **Descripción**
Basándose en el código del notebook `03_refresco_pydantic.ipynb`, deberás crear un **sistema robusto de generación de datos** que demuestre cómo **Pydantic controla y valida las salidas de LLMs**, evitando errores comunes y garantizando la calidad de los datos generados.

## **Requisitos**

### **Variables de Entorno** 
```bash
OPENAI_API_KEY=tu_clave_openai  # Para generación con LLM
HUGGINGFACE_API_KEY=tu_clave_hf  # Alternativa gratuita
```

### **Arquitectura de Clases a Implementar**

1. **SystemConfig** (1 pt) - Configuración y verificación del entorno
   - `verify_pydantic_environment()` - Verificar versión de Pydantic v2
   - `setup_llm_for_structured_output()` - Configurar LLM para salida estructurada

2. **EnhancedCustomer** (2 pts) - Modelo de cliente con validaciones avanzadas
   - Validación de email con dominio empresarial
   - Validación de edad con rangos específicos
   - Validación de crédito con límites por tipo de cliente
   - Normalización automática de datos
   - Validadores personalizados para reglas de negocio

3. **AdvancedTransaction** (2.5 pts) - **MODELO CRÍTICO** con validaciones complejas
   - Validación de ID con formato específico
   - Validación de montos según moneda (JPY sin decimales, EUR/USD con 2)
   - Validación cruzada entre método de pago y moneda
   - Validación de fechas (no futuras, dentro de ventana válida)
   - Validación de métodos de pago con constraints específicos

4. **DocumentProcessor** (2 pts) - Procesador de documentos con validación estricta
   - Validación de tipos de documento permitidos
   - Validación de entidades extraídas con confidence mínimo
   - Validación de coherencia entre tipo de documento y entidades
   - Sanitización de contenido de documentos

5. **LLMDataValidator** (2.5 pts) - **SERVICIO CRÍTICO** de validación de salidas LLM
   - `validate_llm_output()` - Validación estricta de respuestas LLM
   - `retry_with_validation()` - Reintento automático si validación falla
   - `quality_check()` - Verificación de calidad de datos generados
   - Métricas de éxito/fallo de validación

6. **SyntheticDataGenerator** (2 pts) - Generador con control de calidad
   - Generación controlada con múltiples intentos
   - Validación en tiempo real de datos generados
   - Estadísticas de calidad y errores
   - Fallback a datos determinísticos si LLM falla

> **🔥 CONTROL DE LLM**: El sistema debe demostrar cómo Pydantic **previene errores comunes** de LLMs y **garantiza la calidad** de los datos generados mediante validación estricta.

## **Casos de Prueba**

Tu sistema debe manejar estos escenarios:

```python
# Test 1: Validación exitosa de datos LLM
"Generar 3 clientes empresariales con validación completa"

# Test 2: Manejo de errores de validación LLM
"Generar transacciones con validación estricta y retry automático"

# Test 3: Validación cruzada compleja
"Generar transacciones JPY sin decimales y EUR con 2 decimales"

# Test 4: Control de calidad de documentos
"Procesar análisis de documentos con confidence mínimo 0.8"

# Test 5: Estadísticas de validación
"Mostrar métricas de éxito/fallo de validación LLM"
```

## **Validaciones Específicas a Implementar**

### **EnhancedCustomer:**
- ✅ Email debe ser de dominio empresarial (.com, .org, .es, etc.)
- ✅ Edad entre 18-65 años para clientes activos
- ✅ Crédito máximo según tipo: Premium (€5000), Standard (€2000), Basic (€500)
- ✅ Normalización automática de nombres (Title Case)
- ✅ Validación de código de referido (formato específico)

### **AdvancedTransaction:**
- ✅ ID formato: "TXN_YYYYMMDD_XXXXXXXX" (fecha + 8 chars)
- ✅ Montos: JPY sin decimales, otras monedas con 2 decimales exactos
- ✅ Fechas: no futuras, máximo 30 días atrás
- ✅ Validación cruzada: tarjetas no válidas para crypto
- ✅ Límites por moneda: EUR/USD max €10000, JPY max ¥1000000

### **DocumentProcessor:**
- ✅ Tipos permitidos: invoice, contract, report, email solamente
- ✅ Entidades con confidence mínimo 0.7
- ✅ Coherencia: facturas deben tener entidades de tipo "amount"
- ✅ Sanitización: eliminar caracteres especiales del contenido

## **Ejercicios Adicionales**

### **Ejercicio A: Validación de APIs Externas** (1 pt extra)
- Crear modelos para validar respuestas de APIs reales
- Manejo de errores de validación con fallbacks
- Transformación de datos entre formatos

## **Evaluación**

- **Modelos Pydantic Avanzados (6 pts)**: EnhancedCustomer + AdvancedTransaction + DocumentProcessor
- **Control de LLM (2.5 pts)**: LLMDataValidator con retry y quality check
- **Generación Controlada (2.5 pts)**: SyntheticDataGenerator con estadísticas
- **Calidad (1 pt)**: Código limpio, validaciones robustas, manejo de errores
- **Extras (1 pts)**: Ejercicios adicionales implementados

### **Criterios Específicos de Validación:**
- ✅ **Validadores personalizados** con lógica de negocio compleja (2 pts)
- ✅ **Validación cruzada** entre campos relacionados (1.5 pts)
- ✅ **Control de calidad LLM** con retry automático (2 pts)
- ✅ **Métricas y estadísticas** de validación en tiempo real (1.5 pts)

## **Entrega**
Completa el archivo `template.py` con tu implementación funcional y los ejercicios adicionales que elijas.

## **Recursos Adicionales**

### **Documentación Técnica:**
- **Pydantic v2 Docs**: https://docs.pydantic.dev/latest/
- **Validators Guide**: https://docs.pydantic.dev/latest/concepts/validators/
- **Field Constraints**: https://docs.pydantic.dev/latest/concepts/fields/
- **LlamaIndex Structured Output**: https://docs.llamaindex.ai/en/stable/module_guides/querying/structured_outputs/

> Si bien nosotros no hemos usado esta implementación, usar `llm.as_structured_llm(output_cls=Album)` sería también totalmente válido siempre y cuando conserve la funcionalidad exigida en el ejercicio.

¡Demuestra el poder de Pydantic para controlar LLMs y garantizar calidad de datos! 🚀