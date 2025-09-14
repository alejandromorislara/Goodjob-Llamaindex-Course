# 🌟 Programa de Voluntariado Corporativo **#IMPACT**

![Logo GoodJob](https://www.fundaciongoodjob.org/wp-content/smush-webp/2024/01/Logo-Fundacion-183x60_2-scaled.jpg.webp)

Este repositorio contiene el material formativo del **Programa IMPACT#IA** de la **Fundación GoodJob**, una organización sin ánimo de lucro dedicada a fomentar la **inclusión laboral de personas con discapacidad** mediante la **tecnología**.

---

## 🚀 Sobre el Programa IMPACT#IA

### 🎯 Objetivo
Capacitar a los alumnos para la **empleabilidad en roles tecnológicos**, desarrollando competencias en el uso de la **Inteligencia Artificial** y en herramientas clave del ecosistema actual.

---

## 📅 Agenda del Programa

| Sesión | Fecha | Contenido Principal |
|--------|-------|---------------------|
| 📌 **Sesión 1** | 15 septiembre | - Introducción a **llama-index**<br>- Introducción a **LangChain** y **LangGraph**<br>- Otros frameworks (Agno, etc.) |
| 🎯 **Sesión 2** *(este repositorio)* | 16 septiembre | - **Uso de APIs** con llama-index<br>- **Agentes** con llama-index<br>- **Refresco Pydantic**<br>- **Parseo documental** con IA |
| 📌 **Sesión 3** | 17 septiembre | - **Automatización con N8N**<br>- N8N + llama-index (cadena de agentes)<br>- Modelo **MCP/A2A** |
| 📌 **Sesión 4** | 18 septiembre | - Agentes avanzados con más **@tools**<br>- Ampliación del parseo con **Pydantic** |

---

## 🗂️ Estructura del Repositorio

```bash
curso-llamaindex-pydantic/
├── 📓 notebooks/                    # Material principal en Jupyter Notebooks
│   ├── 01_uso_apis_llamaindex.ipynb
│   ├── 02_agentes_llamaindex.ipynb
│   ├── 03_refresco_pydantic.ipynb
│   ├── 04_parseo_documentos.ipynb
│   └── sources/                     # Recursos visuales (diagramas, imágenes)
├── 🏋️ exercises/                    # Ejercicios prácticos
│   ├── bloque1_apis/               # Ejercicio de APIs con NewsAPI
│   ├── bloque1_agentes/            # Ejercicio de agentes multi-sistema
│   ├── bloque2_pydantic/           # Ejercicio de validación avanzada
│   └── bloque2_parseo/             # Ejercicio de parseo web con LLM
│       ├── enunciado.md
│       └─ template.py
├── ✅ solutions/                    # Soluciones completas de los ejercicios
│   ├── bloque1_apis_sol.py
│   ├── bloque1_agentes_sol.py
│   ├── bloque2_pydantic_sol.py
│   └── bloque2_parseo_sol.py
├── 🔧 src/                         # Código fuente del curso
│   ├── apis/                       # APIs integradas (NewsAPI, WeatherAPI)
│   ├── embeddings/                 # Modelos de embeddings
│   └── string_utils.py             # Utilidades de procesamiento
├── 🏗️ models/                      # Modelos Pydantic de ejemplo
│   ├── earnings_model.py
│   └── radiohead.py
├── 📊 data/                        # Datasets de ejemplo
│   ├── radiohead.json             # Dataset principal para parseo
│   ├── BOE-A-1978-31229-consolidado.pdf
│   
├── 🗄️ chroma_db/                   # Base de datos vectorial ChromaDB
├── 🕷️ llamaindex_docs_crawler/      # Web scraper para documentación
├── requirements.txt                # Dependencias Python
└── setup_instructions.md           # Guía de configuración detallada
````

---

## 🎯 Ejercicios Prácticos

### 🔧 Bloque 1: APIs y Agentes

#### 📰 **Ejercicio APIs**: Agente de Noticias

* **Objetivo**: Implementar un agente que busque noticias en tiempo real
* **Tecnologías**: NewsAPI, LlamaIndex, agentes conversacionales
* **Funcionalidades**: Configuración LLM, integración de herramientas, memoria conversacional

#### 🤖 **Ejercicio Agentes**: Sistema Multi-Agente

* **Objetivo**: Diseñar un sistema multi-agente con arquitectura orientada a objetos
* **Tecnologías**: NewsAPI, WeatherAPI, ChromaDB, deduplicación con LLM
* **Funcionalidades**: NewsCache, DeduplicationService, AgentThoughtManager

---

### 🔧 Bloque 2: Pydantic y Parseo

#### ✅ **Ejercicio Pydantic**: Validación Avanzada

* **Objetivo**: Sistema de validación y generación de datos sintéticos
* **Tecnologías**: Pydantic v2, validaciones personalizadas, control de LLMs
* **Funcionalidades**: EnhancedCustomer, AdvancedTransaction, LLMDataValidator

#### 📄 **Ejercicio Parseo**: Extracción Web y Base de Datos Vectorial

* **Objetivo**: Sistema completo de parseo JSON + extracción web con LLM tipado
* **Tecnologías**: SimpleWebPageReader, LLM tipado, ChromaDB, Pydantic v2
* **Funcionalidades**: Extracción de letras desde web, análisis de sentimientos, persistencia vectorial

---

## 🛠️ Configuración del Entorno

### 📋 Requisitos Previos

* **Python 3.12+**
* **Entorno virtual** (recomendado)
* **API keys** para servicios externos

### 🔑 Variables de Entorno

```bash
# OpenAI API Key (OBLIGATORIO para ejercicios con LLM)
OPENAI_API_KEY=sk-tu_api_key_de_openai

# Hugging Face Token (OBLIGATORIO para embeddings)
HF_TOKEN=hf_tu_token_aqui

# APIs opcionales para ejercicios específicos
OPENWEATHER_KEY=tu_api_key_de_openweather
NEWS_API_KEY=tu_api_key_de_newsapi
```

### 🚀 Instalación Rápida

```bash
# Clonar el repositorio
git clone https://github.com/alejandromorislara/Goodjob-Llamaindex-Course
cd curso-llamaindex-pydantic

# Crear entorno virtual
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate      # Windows

# Instalar dependencias
pip install -r requirements.txt

# Configurar variables de entorno
cp .env.example .env
# Editar .env con tus API keys
```

---

## 🚀 Cómo Empezar

1. **📖 Lee la guía de configuración** → `setup_instructions.md`
2. **🔧 Configura tu entorno** con las APIs necesarias (especialmente `OPENAI_API_KEY`)
3. **📓 Explora los notebooks** en orden secuencial:
   - `01_uso_apis_llamaindex.ipynb` - Fundamentos de APIs
   - `02_agentes_llamaindex.ipynb` - Sistemas multi-agente
   - `03_refresco_pydantic.ipynb` - Validación avanzada
   - `04_parseo_documentos.ipynb` - Parseo con LLM
4. **🏋️ Completa los ejercicios** de cada bloque:
   - Bloque 1: APIs y Agentes
   - Bloque 2: Pydantic y Parseo Web
5. **✅ Revisa las soluciones** para comparar enfoques

---

## 🤝 Contribuir y Contacto

<table>
<tr>
<td style="width:70%; vertical-align:top;">

### 🙌 Cómo participar como voluntario formador

Si deseas sumarte al programa **#IMPACT** como voluntario:

1. **📧 Contacta** con la coordinación del programa  
2. **📅 Confirma** tu disponibilidad en el calendario de sesiones  
3. **📚 Prepara** tu sesión con los materiales y contenidos asignados  
4. **🎯 Comparte** tu experiencia y conocimiento con los participantes  

</td>
<td style="width:30%; text-align:center;">

<img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRGCrL-V9c0NgujafmRoO5ZIL-92l0GF5LvMA&s" width="180"/>

</td>
</tr>
</table>

---

### 🏢 Dirección de la Fundación
**Sector Oficios 32, 28760 Tres Cantos (Madrid)**  

---

### 🌐 Más información
[🌍 www.fundaciongoodjob.org](https://www.fundaciongoodjob.org)

---

![Fundación GoodJob](https://www.mercanza.es/xen_media/blog-goodjob.jpg)


---

*Juntos construimos un futuro más inclusivo gracias a la tecnología y la formación especializada.*

---

**© 2025 Fundación GoodJob - Programa de Voluntariado Corporativo #IMPACT**

