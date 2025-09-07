# 🌐 Ejercicio: Sistema Multi-Agente con APIs de Noticias y Clima - Arquitectura con Clases

## **Objetivo**
Implementar un sistema multi-agente profesional utilizando **arquitectura orientada a objetos** que integre APIs externas (News API y Weather API) con deduplicación inteligente LLM y gestión avanzada de agentes especializados.

## **Descripción**
Basándose en los conceptos aprendidos en el notebook `02_agentes_llamaindex.ipynb`, deberás crear un **ecosistema de clases especializadas** que trabajen coordinadamente para proporcionar información actualizada sobre noticias y clima, con **deduplicación individual por artículo usando LLM** y **gestión de pensamientos de agentes**.

## **Requisitos**

### **Variables de Entorno** 
```bash
OPENAI_API_KEY=tu_clave_openai  # Recomendado para mejor rendimiento
HUGGINGFACE_API_KEY=tu_clave_hf  # Alternativa gratuita
NEWS_API_KEY=tu_clave_de_newsapi_org  # Para herramientas de noticias
WEATHER_API_KEY=tu_clave_openweather  # Para herramientas de clima
```

### **Arquitectura de Clases a Implementar**

1. **SystemConfig** (1 pt) - Configuración global del sistema
   - `verify_environment()` - Verificar dependencias instaladas
   - `setup_llm_environment()` - Configurar LLM y embeddings OpenAI

2. **NewsCache** (2 pts) - Gestión del cache vectorial de noticias
   - `initialize()` - Configurar ChromaDB con colección persistente
   - `search()` - Búsqueda semántica en cache
   - `search_similar_articles_by_title()` - Buscar artículos similares por título
   - `insert_article()` - Insertar artículo individual en cache

3. **DeduplicationService** (2.5 pts) - **SERVICIO CRÍTICO** de deduplicación LLM
   - `compare_article_with_existing()` - Comparación LLM individual artículo vs BBDD
   - `process_articles()` - Procesamiento completo con decisiones tipadas
   - **FLUJO**: API → Buscar similares por título → LLM evalúa → Decisión INSERT/SKIP

4. **AgentThoughtManager** (1 pt) - Gestión de pensamientos de agentes
   - `save_thought()` - Guardar decisiones y razonamientos de agentes
   - `get_recent_thoughts()` - Consultar historial de pensamientos

5. **NewsService** (1.5 pts) - Servicio de noticias con deduplicación
   - `search_news_with_deduplication()` - Búsqueda con deduplicación automática LLM

6. **WeatherService** (1.5 pts) - Servicio meteorológico con alertas
   - `calculate_fire_risk_index()` - Cálculo de riesgo de incendio con reglas lógicas
   - `get_weather_with_alerts()` - Clima actual con alertas automáticas

7. **MultiAgentSystem** (1.5 pts) - Sistema principal coordinador
   - `initialize()` - Inicialización completa del sistema
   - `_create_agents()` - Creación de NewsAgent, WeatherAgent, RouterAgent
   - `_create_workflow()` - Configuración de AgentWorkflow con handoffs
   - `run_tests()` - Ejecución de casos de prueba

> **🔥 DEDUPLICACIÓN LLM**: El sistema debe evaluar **individualmente cada artículo** comparando su descripción con artículos similares en BBDD usando LLM para decidir INSERT/SKIP.

## **Casos de Prueba**

Tu sistema debe manejar estas consultas:

```python
# Test 1: NewsAgent - Primera consulta (busca en API, evalúa con LLM, inserta únicos)
"¿Cuáles son las últimas noticias sobre inteligencia artificial?"

# Test 2: NewsAgent - Segunda consulta similar (LLM detecta duplicados y los omite)
"Busca información sobre inteligencia artificial en las noticias"

# Test 3: WeatherAgent - Consulta simple con alertas automáticas
"¿Qué tiempo hace en Madrid hoy?"

# Test 4: WeatherAgent - Función de riesgo de incendio
"¿Cuál es el índice de confort térmico en Barcelona con 25°C y 60% humedad?"

# Test 5: NewsAgent - Consulta específica con deduplicación
"¿Hay noticias sobre tormentas en España?"
```

## **Ejercicios Adicionales**

### **Ejercicio A: Análisis de Sentimientos en Noticias** (1 pt extra)
- Implementar análisis de sentimientos en NewsAgent
- Clasificar noticias como positivas, negativas o neutrales
- Generar resúmenes con análisis emocional

### **Ejercicio B: Sistema de Alertas Meteorológicas** (1 pt extra)
- Crear función `evaluar_riesgo_meteorologico()`
- Detectar condiciones extremas automáticamente
- Generar alertas personalizadas por ubicación

### **Ejercicio C: Integración con Llamatrace** (1 pt extra)
- Integra tu flujo multi-agente con [**Llamatrace**](https://phoenix.arize.com/llamatrace/)



## **Evaluación**

- **Arquitectura con Clases (10 pts)**: SystemConfig + NewsCache + DeduplicationService + AgentThoughtManager + NewsService + WeatherService + MultiAgentSystem
- **Calidad (1 pt)**: Código limpio, separación de responsabilidades, manejo de errores
- **Extras (3 pts)**: Ejercicios adicionales implementados

### **Criterios Específicos de Deduplicación LLM:**
- ✅ **DeduplicationService completo** con comparación LLM individual (2 pts)
- ✅ **Modelos Pydantic tipados** (ArticleComparisonResult, DeduplicationResult) (0.5 pts)
- ✅ **AgentThoughtManager** guardando pensamientos con reasoning (0.5 pts)
- ✅ **Flujo correcto**: API → Buscar similares → LLM evalúa → INSERT/SKIP (2 pts)

## **Entrega**
Completa el archivo `template.py` con tu implementación funcional y los ejercicios adicionales que elijas.

## **Recursos Adicionales**

### **APIs Requeridas:**
- **NewsAPI**: https://newsapi.org/ (500 requests/día gratuitos)
- **OpenWeatherMap**: https://openweathermap.org/ (1000 requests/día gratuitos)
- **HuggingFace**: https://huggingface.co/ (para embeddings alternativos)

### **Documentación Técnica:**
- **News API Docs**: https://newsapi.org/docs
- **OpenWeather API Docs**: https://openweathermap.org/api
- **LlamaIndex Agent Workflows**: https://docs.llamaindex.ai/en/stable/module_guides/workflow/

