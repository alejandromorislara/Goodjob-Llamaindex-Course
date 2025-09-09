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
├── 📓 notebooks/         # Material principal en Jupyter Notebooks
│   ├── 01_uso_apis_llamaindex.ipynb
│   ├── 02_agentes_llamaindex.ipynb
│   ├── 03_refresco_pydantic.ipynb
│   ├── 04_parseo_documentos.ipynb
│   ├── extra_retos.ipynb
│   └── sources/          # Recursos visuales
├── 🏋️ exercises/         # Ejercicios prácticos
│   ├── bloque1_apis/
│   ├── bloque1_agentes/
│   ├── bloque2_pydantic/
│   └── bloque2_parseo/
├── ✅ solutions/         # Soluciones de los ejercicios
├── 🎯 final_proyect/     # Proyecto final
├── 🔧 src/               # Código fuente del curso
│   ├── apis/
│   ├── embeddings/
│   └── parsers/
├── 📊 data/              # Datos de ejemplo
├── 🗄️ chroma_db/          # Base de datos vectorial
├── requirements.txt       # Dependencias Python
└── setup_instructions.md  # Guía de configuración
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

#### 📄 **Ejercicio Parseo**: Extracción Documental

* **Objetivo**: Parseo de documentos con agentes especializados
* **Tecnologías**: PDF parsing, HTML parsing, Pydantic models
* **Funcionalidades**: Extracción estructurada, validación de contenido

---

## 🎮 Proyecto Final

*La información sobre el proyecto final se añadirá próximamente.*

---

## 🛠️ Configuración del Entorno

### 📋 Requisitos Previos

* **Python 3.12+**
* **Entorno virtual** (recomendado)
* **API keys** para servicios externos

### 🔑 Variables de Entorno

```bash
# Hugging Face Token (OBLIGATORIO)
HF_TOKEN=hf_tu_token_aqui

# APIs opcionales
OPENWEATHER_KEY=tu_api_key_de_openweather
NEWS_API_KEY=tu_api_key_de_newsapi

# PokeAPI (ya configurado)
POKEAPI_BASE_URL=https://pokeapi.co/api/v2/
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
2. **🔧 Configura tu entorno** con las APIs necesarias
3. **📓 Explora los notebooks** en orden secuencial
4. **🏋️ Completa los ejercicios** de cada bloque
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

