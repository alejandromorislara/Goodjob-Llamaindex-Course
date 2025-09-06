# 🌐 Ejercicio: Sistema Multi-Agente con APIs de Noticias y Clima

## **Objetivo**
Implementar un sistema multi-agente especializado que integre APIs externas (News API y Weather API) con un router inteligente y funcionalidades avanzadas de procesamiento de datos meteorológicos.

## **Descripción**
Basándose en los conceptos aprendidos en el notebook `02_agentes_llamaindex.ipynb`, deberás crear un ecosistema de agentes especializados que trabajen con APIs externas para proporcionar información actualizada sobre noticias y clima, incluyendo análisis avanzado de tendencias meteorológicas.

## **Requisitos**

### **Variables de Entorno** 
```bash
OPENAI_API_KEY=tu_clave_openai  # Recomendado para mejor rendimiento
HUGGINGFACE_API_KEY=tu_clave_hf  # Alternativa gratuita
NEWS_API_KEY=tu_clave_de_newsapi_org  # Para herramientas de noticias
WEATHER_API_KEY=tu_clave_openweather  # Para herramientas de clima
```

### **Funcionalidades a Implementar**

1. **Sistema de Cache de Noticias con ChromaDB** (2.5 pts)
   - Configurar ChromaDB con colección persistente para noticias
   - Implementar búsqueda semántica en noticias guardadas ANTES de llamar a la API
   - Guardar automáticamente nuevas noticias obtenidas de la API

2. **Agente de Noticias (NewsAgent)** (2.5 pts)
   - Integración con News API para búsqueda de noticias
   - Filtrado por categorías, fechas y fuentes
   - Sistema híbrido: ChromaDB → API si no encuentra resultados

3. **Agente Meteorológico (WeatherAgent)** (2.5 pts)
   - Integración con Weather API para datos actuales
   - Función adicional: `calcular_indice_riesgo_incendio()` 
   - Predicciones y alertas meteorológicas

> **💡 Pista** : Para `calcular_indice_riesgo_incendio()` puedes usar reglas lógicas de temperatura!

4. **Router Inteligente** (1.5 pts)
   - Clasificación automática de consultas (noticias vs clima)
   - Enrutamiento a agentes especializados
   - Manejo de consultas híbridas

5. **Sistema de Coordinación** (1 pts)
   - Detección de consultas que requieren ambos agentes
   - Combinación inteligente de resultados
   - Gestión del estado entre agentes

## **Casos de Prueba**

Tu sistema debe manejar estas consultas:

```python
# Test 1: Primera consulta - NewsAgent (debe buscar en API y guardar en ChromaDB)
"¿Cuáles son las últimas noticias sobre inteligencia artificial?"

# Test 2: Segunda consulta similar - NewsAgent (debe encontrar en ChromaDB)
"Busca información sobre inteligencia artificial en las noticias"

# Test 3: Consulta simple - WeatherAgent  
"¿Qué tiempo hace en Madrid hoy?"

# Test 4: Función personalizada - WeatherAgent
"¿Cuál es el índice de confort térmico en Barcelona con 25°C y 60% humedad?"

# Test 5: Consulta compleja - Coordinación
"¿Hay noticias sobre tormentas en España y cómo está el tiempo en las ciudades afectadas?"

# Test 6: Verificación de cache - NewsAgent
"Muéstrame noticias relacionadas con IA que ya hayas buscado antes"
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

- **Funcional (10 pts)**: NewsAgent + WeatherAgent + Router + Coordinación
- **Calidad (1 pt)**: Código limpio, manejo de errores, documentación
- **Extras (3 pts)**: Ejercicios adicionales implementados

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

