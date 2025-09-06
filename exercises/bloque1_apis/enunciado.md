# 🤖 Ejercicio: Agente de Noticias con LlamaIndex

## **Objetivo**
Implementar un agente de IA que pueda buscar noticias en tiempo real y mantener conversaciones contextuales usando la API de NewsAPI.

## **Descripción**
Basándose en los conceptos aprendidos en el notebook `01_uso_apis_llamaindex.ipynb`, deberás crear un agente conversacional que integre la herramienta `news_search_tool()` de `news_api.py` para buscar noticias actualizadas.

## **Requisitos**

### **Variables de Entorno** 
```bash
NEWS_API_KEY=tu_clave_de_newsapi_org
OPENAI_API_KEY=tu_clave_openai  # Opcional
HUGGINGFACE_API_KEY=tu_clave_hf  # Para modelo gratuito
```

### **Funcionalidades a Implementar**

1. **Configuración del LLM** (2.5 pts)
   - Configurar un modelo usando OpenAI o HuggingFace
   - Verificar funcionamiento básico

2. **Integración de la Herramienta** (2.5 pts)
   - Importar y usar `news_search_tool()` de `src.apis.news_api`
   - Crear AgentWorkflow con la herramienta de noticias

3. **Memoria Conversacional** (2.5 pts)
   - Implementar contexto usando `Context` de LlamaIndex
   - El agente debe recordar temas y preferencias del usuario

4. **Conversación Inteligente** (2.5 pts)
   - Mostrar herramientas en uso durante la búsqueda
   - Manejo de errores y respuestas naturales

## **Casos de Prueba**

Tu agente debe manejar estas conversaciones:

```python
# Test 1: Búsqueda básica
"Busca noticias sobre inteligencia artificial"

# Test 2: Memoria + contexto
"Mi nombre es Carlos. Busca noticias sobre OpenAI en español"

# Test 3: Recordar preferencias  
"¿Recuerdas mi nombre? Busca más noticias sobre IA"

# Test 4: Búsqueda específica
"Busca 5 noticias recientes sobre tecnología ordenadas por relevancia"
```

## **Evaluación**

- **Funcional (8 pts)**: LLM + herramienta + agente + memoria
- **Calidad (1 pt)**: Código limpio y manejo de errores  
- **Extra (1 pt)**: Mejoras adicionales (filtros, formato, multiidioma)

## **Entrega**
Completa el archivo `template.py` con tu implementación funcional.
