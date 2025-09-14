# 🎵 Ejercicio: Parseo Avanzado con Extracción Web y Base de Datos Vectorial

## **Objetivo**
Implementar un sistema completo de **parseo de documentos JSON** que utilice **SimpleWebPageReader** y **LLM tipado** para extraer letras desde páginas web, analizar sentimientos de canciones, validar toda la información con **Pydantic**, y persistir los datos en una **base de datos vectorial** para búsquedas semánticas avanzadas.

## **Descripción**
Basándose en el código del notebook `04_parseo_documentos.ipynb` y utilizando el archivo `data/radiohead.json`, deberás crear un **sistema avanzado de análisis musical** que combine parseo de documentos, **extracción web de letras con SimpleWebPageReader**, **análisis de sentimientos con LLM tipado**, y **persistencia en base de datos vectorial** para búsquedas semánticas inteligentes.

**NOTA IMPORTANTE**: Para la extracción de letras se recomienda usar un modelo de IA más potente que gpt-4o-mini. Para la extracción de texto sería recomendable usar técnicas de web scraping como se comentó en el notebook y durante el curso.

## **Contexto del Dataset**
El archivo `radiohead.json` contiene información estructurada en formato JSON-LD (Schema.org) sobre la banda Radiohead, incluyendo:
- **Información del artista**: nombre, género, descripción, imagen
- **Álbumes**: con metadatos completos (fecha, editorial, número de tracks)
- **Canciones**: con duración, género, álbum asociado, URLs

## **Requisitos**

### **Variables de Entorno** 
```bash
OPENAI_API_KEY=tu_clave_openai  # Para procesamiento con LLM
```

### **Arquitectura de Clases a Implementar**

1. **LyricsAnalysis** (1.5 pts) - **MODELO CRÍTICO** para análisis de letras
   - Validación de letras no vacías
   - Análisis de sentimiento (7 tipos: positive, negative, neutral, melancholic, energetic, romantic, angry)
   - Detección de idioma automática
   - Extracción de temas principales (máximo 5)
   - Cálculo de intensidad emocional (0.0-1.0)

2. **RadioheadParser** (2 pts) - Parser principal del documento JSON
   - `load_json()` - Cargar y validar estructura del JSON
   - `extract_artist_info()` - Extraer información del artista
   - `extract_albums()` - Extraer álbumes únicos con metadatos
   - `extract_songs()` - Extraer canciones con información completa

3. **Album** (1.5 pts) - Modelo Pydantic para álbumes
   - Validación de fechas de publicación (1900-2030)
   - Validación de número de tracks (1-50)
   - Normalización de nombres y géneros
   - Validación de URLs y metadatos

4. **Song** (2 pts) - Modelo Pydantic para canciones con análisis
   - Validación de duración ISO 8601 (PT3M55S)
   - Conversión automática a formato legible (3:55)
   - Integración con análisis de letras (LyricsAnalysis)
   - Validación de relaciones álbum-canción

5. **Artist** (1 pt) - Modelo Pydantic para artista
   - Validación de géneros musicales permitidos
   - Validación de URLs de imagen
   - Normalización de descripciones

6. **WebLyricsExtractor** (3 pts) - **SERVICIO CRÍTICO** de extracción web con LLM
   - `extract_lyrics_from_url()` - Extraer letras desde URLs usando SimpleWebPageReader + LLM tipado
   - `analyze_lyrics_with_web_data()` - Análisis de sentimiento con datos web
   - `process_song_with_web_url()` - Procesamiento completo de canción con URL
   - `batch_extract_lyrics_from_web()` - Procesamiento en lote desde web
   - Integración con SimpleWebPageReader y OpenAI
   - Validación automática con Pydantic

7. **VectorMusicDatabase** (4 pts) - **BASE DE DATOS VECTORIAL** persistente (ÚNICA BD)
   - `initialize_database()` - Configurar ChromaDB persistente
   - `save_songs_to_vector_db()` - Guardar canciones con embeddings y metadata completa
   - `search_songs_by_sentiment()` - Búsqueda por sentimiento
   - `search_songs_by_lyrics()` - Búsqueda semántica por letras
   - `search_songs_by_album()` - Búsqueda por álbum específico
   - `search_songs_by_themes()` - Búsqueda por temas musicales
   - `get_all_songs()` - Obtener todas las canciones almacenadas
   - `get_statistics()` - Estadísticas completas de sentimientos, álbumes y temas
   - `delete_all_songs()` - Limpieza para testing

8. **LlamaIndexProcessor** (2 pts) - Procesador con LlamaIndex
   - `create_documents()` - Convertir datos a documentos LlamaIndex
   - `create_index()` - Crear índice vectorial
   - `query_music_info()` - Consultas inteligentes

> **🔥 FLUJO COMPLETO**: El sistema debe demostrar el pipeline completo: JSON → SimpleWebPageReader → LLM Tipado → Pydantic → ChromaDB Vectorial

## **Casos de Prueba**

Tu sistema debe manejar estos escenarios:

```python
# Test 1: Parseo completo del JSON
"Cargar radiohead.json y extraer toda la información estructurada"

# Test 2: Extracción web de letras con SimpleWebPageReader y LLM tipado
"Extraer letras desde URLs de letras.com usando SimpleWebPageReader"
"Procesar contenido web con LLM tipado para obtener letras, visualizaciones y compositores"
"Validar que el análisis incluye: letras, sentimiento, idioma, temas, intensidad, visualizaciones"

# Test 3: Persistencia en base de datos vectorial (ÚNICA BD)
"Guardar canciones con letras en ChromaDB con embeddings automáticos"
"Almacenar toda la metadata (artista, álbum, sentimiento, temas) en ChromaDB"
"No usar SQL - solo base de datos vectorial persistente"

# Test 4: Búsquedas semánticas avanzadas
"Buscar canciones melancólicas de Radiohead"
"Encontrar canciones que hablen de amor y relaciones"
"Filtrar por intensidad emocional alta (>0.7)"
"Buscar por álbum específico (ej: 'OK Computer')"
"Buscar por temas musicales (ej: 'existencialismo', 'alienación')"

# Test 5: Estadísticas completas de la base de datos vectorial
"Mostrar distribución de sentimientos en la discografía"
"Identificar temas más comunes en las letras"
"Analizar evolución emocional por álbum"
"Mostrar estadísticas de idiomas detectados"
"Calcular intensidad emocional promedio"
```

## **Validaciones Específicas a Implementar**

### **LyricsAnalysis:**
- ✅ Letras no vacías y sanitizadas
- ✅ Sentimiento en enum válido (7 tipos disponibles)
- ✅ Idioma detectado automáticamente
- ✅ Máximo 5 temas principales
- ✅ Intensidad emocional entre 0.0-1.0
- ✅ Conteo automático de palabras

### **Album:**
- ✅ Fecha de publicación entre 1900-2030
- ✅ Número de tracks entre 1-50
- ✅ Nombre no vacío y normalizado (Title Case)
- ✅ URL válida y accesible
- ✅ Editorial no vacía

### **Song:**
- ✅ Duración en formato ISO 8601 válido (PT\d+M\d+S)
- ✅ Conversión automática a formato MM:SS
- ✅ Integración obligatoria con LyricsAnalysis
- ✅ Relación válida con álbum existente
- ✅ URL única y válida

### **Artist:**
- ✅ Género en lista permitida: ["Rock Alternativo", "Rock", "Electronic", "Experimental"]
- ✅ Descripción sanitizada (sin caracteres especiales)
- ✅ URL de imagen válida
- ✅ Nombre único y normalizado

### **VectorMusicDatabase:**
- ✅ Persistencia en ChromaDB con configuración correcta
- ✅ Embeddings automáticos para búsquedas semánticas
- ✅ Metadata estructurada para filtros avanzados
- ✅ Índices optimizados para consultas por sentimiento
- ✅ Búsquedas híbridas (semántica + filtros)

## **Estructura de Base de Datos Vectorial**

### **ChromaDB Collection Schema**
```python
# Colección principal: "radiohead_songs"
collection_metadata = {
    "description": "Radiohead songs with lyrics and sentiment analysis",
    "version": "1.0",
    "embedding_model": "text-embedding-ada-002"
}

# Estructura de documentos
document_structure = {
    "text": "Letras completas + metadata de la canción",
    "metadata": {
        "song_id": "ID único de la canción",
        "song_name": "Nombre de la canción",
        "artist_name": "Radiohead",
        "album_name": "Nombre del álbum",
        "genre": "Género musical",
        "duration": "Duración formateada (MM:SS)",
        "sentiment": "Sentimiento principal (enum)",
        "language": "Idioma detectado",
        "emotional_intensity": "Intensidad emocional (0.0-1.0)",
        "word_count": "Número de palabras en letras",
        "themes": "Temas principales separados por comas",
        "url": "URL original de la canción"
    },
    "id": "song_{song_id}"
}
```

### **Índices y Búsquedas Soportadas**
- 🔍 **Búsqueda semántica**: Por contenido de letras usando embeddings
- 🎭 **Filtro por sentimiento**: Canciones melancólicas, energéticas, etc.
- 🌍 **Filtro por idioma**: Inglés, español, etc.
- 📊 **Filtro por intensidad**: Rango de intensidad emocional
- 🏷️ **Filtro por temas**: Amor, política, existencialismo, etc.
- 💿 **Filtro por álbum**: Búsquedas específicas por disco

## **Ejercicios Adicionales**

### **Ejercicio A: Análisis Musical con LLM** (1 pt extra)
- Usar LlamaIndex para generar análisis de la discografía
- Identificar patrones en duraciones y géneros
- Generar recomendaciones basadas en el catálogo

### **Ejercicio B: Exportación de Datos** (0.5 pts extra)
- Exportar datos a CSV/JSON con formato específico
- Crear reportes de estadísticas musicales
- Generar playlist automáticas por duración

## **Evaluación**

- **Modelos Pydantic (4.5 pts)**: WebLyricsData + LyricsAnalysis + RadioheadParser + Album + Song + Artist
- **Extracción Web con LLM (3 pts)**: WebLyricsExtractor con SimpleWebPageReader y LLM tipado
- **Base de Datos Vectorial (4 pts)**: VectorMusicDatabase como única BD persistente
- **LlamaIndex (2 pts)**: LlamaIndexProcessor con consultas inteligentes
- **Calidad (1 pt)**: Código limpio, validaciones robustas, manejo de errores
- **Extras (0.5 pts)**: Ejercicios adicionales implementados

### **Criterios Específicos de Validación:**
- ✅ **Parseo robusto** del JSON con manejo de errores (1.5 pts)
- ✅ **Validaciones Pydantic** complejas con reglas de negocio (2 pts)
- ✅ **Extracción Web con LLM tipada** usando SimpleWebPageReader (3 pts)
- ✅ **Base de datos vectorial** como única fuente de persistencia (4 pts)
- ✅ **Búsquedas semánticas** avanzadas con filtros múltiples (2 pts)

## **Entrega**
Completa el archivo `template.py` con tu implementación funcional y los ejercicios adicionales que elijas.

## **Recursos Adicionales**

### **Documentación Técnica:**
- **LlamaIndex SimpleDirectoryReader**: https://docs.llamaindex.ai/en/stable/module_guides/loading/simpledirectoryreader/
- **LlamaIndex Structured Output**: https://docs.llamaindex.ai/en/stable/module_guides/querying/structured_outputs/
- **Pydantic Models**: https://docs.pydantic.dev/latest/concepts/models/
- **Pydantic Field Validators**: https://docs.pydantic.dev/latest/concepts/validators/
- **ChromaDB Documentation**: https://docs.trychroma.com/
- **ChromaDB Python Client**: https://docs.trychroma.com/reference/py-client
- **JSON-LD Schema.org**: https://schema.org/MusicGroup

### **Archivos de Referencia:**
- `data/radiohead.json` - Dataset principal
- `models/radiohead.py` - Ejemplo de parser básico
- `notebooks/04_parseo_documentos.ipynb` - Ejemplos de LlamaIndex

¡Demuestra el poder del parseo inteligente con LlamaIndex, la extracción tipada con LLM, y la persistencia vectorial con ChromaDB! 🎵🚀

### **Puntos Clave del Ejercicio:**
- 🤖 **LLM Tipado**: Extracción de letras y sentimientos con validación Pydantic
- 🔍 **Búsquedas Semánticas**: Consultas inteligentes por contenido y metadata
- 📊 **Análisis Completo**: Sentimientos, temas, intensidad emocional
- 💾 **Persistencia Total**: Toda la información en ChromaDB con embeddings