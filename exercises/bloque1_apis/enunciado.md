# 🌐 Ejercicio: Sistema Multi-Agente con APIs y ChromaDB

## **Objetivo**
Implementar un sistema multi-agente avanzado que integre bases de datos vectoriales (ChromaDB) con múltiples APIs externas, incluyendo un router inteligente y orquestación de agentes especializados.

## **Descripción**
Basándose en los conceptos aprendidos en el notebook `02_agentes_llamaindex.ipynb`, deberás crear un ecosistema de agentes especializados que trabajen en conjunto para resolver consultas complejas que requieren múltiples fuentes de información.

## **Requisitos**

### **Variables de Entorno** 
```bash
OPENAI_API_KEY=tu_clave_openai  # Recomendado para mejor rendimiento
HUGGINGFACE_API_KEY=tu_clave_hf  # Alternativa gratuita
NEWS_API_KEY=tu_clave_de_newsapi_org  # Para herramientas de noticias
WEATHER_API_KEY=tu_clave_openweather  # Para herramientas de clima
```

### **Funcionalidades a Implementar**

1. **Base de Datos Vectorial** (2.5 pts)
   - Configurar ChromaDB con persistencia
   - Cargar y procesar documentos de ejemplo
   - Crear query engine para búsqueda semántica

2. **Agentes Especializados** (2.5 pts)
   - DocsAgent: Especialista en documentos empresariales
   - APIAgent: Experto en herramientas externas (clima, finanzas, noticias)
   - MathAgent: Genio de cálculos matemáticos

3. **Router Inteligente** (2.5 pts)
   - Sistema híbrido: reglas + LLM
   - Enrutamiento automático a agentes apropiados
   - Medición de precisión del router

4. **Orquestación Multi-Agente** (2.5 pts)
   - Detección de consultas complejas
   - Coordinación de múltiples agentes
   - Combinación inteligente de resultados

## **Casos de Prueba**

Tu sistema debe manejar estas consultas:

```python
# Test 1: Consulta simple - DocsAgent
"¿Cuál es la comisión para transferencias SWIFT a Estados Unidos?"

# Test 2: Consulta API - APIAgent  
"¿A cuánto está el cambio EUR/USD hoy?"

# Test 3: Consulta matemática - MathAgent
"Calcula el 15% de 2500 euros"

# Test 4: Consulta compleja - Orquestación
"Si envío 1200 EUR por SWIFT a Estados Unidos, ¿cuánto llega después de comisiones y a qué tipo de cambio?"

# Test 5: Router inteligente
"¿Qué tiempo hace en Madrid?" # Debe ir a APIAgent automáticamente
```

## **Ejercicios Adicionales**

### **Ejercicio A: Evaluación del Router** (1 pt extra)
- Crear dataset de prueba etiquetado
- Medir precisión de reglas vs LLM
- Implementar sistema de confianza

### **Ejercicio B: Agente de Cotización** (1 pt extra)
- Especialista en cálculo de costos de envío
- Integración completa: Docs → Math → API
- Desglose detallado paso a paso

### **Ejercicio C: Observabilidad** (1 pt extra)
- Logging de interacciones en JSONL
- Métricas de latencia y uso
- Reportes de rendimiento

## **Evaluación**

- **Funcional (8 pts)**: ChromaDB + Agentes + Router + Orquestación
- **Calidad (1 pt)**: Código limpio, manejo de errores, documentación
- **Extras (3 pts)**: Ejercicios adicionales implementados

## **Entrega**
Completa el archivo `template.py` con tu implementación funcional y los ejercicios adicionales que elijas.

## **Recursos Adicionales**

### **APIs Gratuitas Recomendadas:**
- **NewsAPI**: https://newsapi.org/ (500 requests/día)
- **OpenWeatherMap**: https://openweathermap.org/ (1000 requests/día)
- **HuggingFace**: https://huggingface.co/ (uso limitado gratuito)

### **Fuentes de Documentos PDF:**
- EUR-Lex / Diario Oficial de la UE
- Oficina de Publicaciones de la UE
- SEC EDGAR (informes financieros)
- arXiv (papers científicos)
- PubMed Central (artículos médicos)

### **Mejores Prácticas:**
1. **Especialización**: Un agente por dominio específico
2. **Validación**: Siempre valida inputs y outputs
3. **Persistencia**: Usa ChromaDB para evitar reprocesamiento
4. **Monitoreo**: Implementa observabilidad desde el inicio
5. **Testing**: Crea datasets de prueba etiquetados