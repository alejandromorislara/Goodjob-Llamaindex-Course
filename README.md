# Programa de Voluntariado Corporativo #IMPACT

![Logo GoodJob](https://www.fundaciongoodjob.org/wp-content/smush-webp/2024/01/Logo-Fundacion-183x60_2-scaled.jpg.webp)

Este repositorio contiene el material formativo para el **Programa IMPACT#IA** de la **Fundación GoodJob**, una organización sin ánimo de lucro dedicada a fomentar la inclusión laboral de personas con discapacidad a través de la tecnología.



## 🚀 Programa IMPACT#IA

### 🎯 Objetivo del Programa
Preparar a los alumnos para la **empleabilidad en distintos roles en sectores tecnológicos**, aportándoles las fortalezas en el uso de la **Inteligencia Artificial**.

### 📅 Programa Detallado por Sesiones

#### 📌 SESIÓN 1 - 15 de septiembre
- **[BLOQUE 1]** Introducción a llama-index
- **[BLOQUE 1]** Introducción a langchain y langgraph
- **[BLOQUE 2]** Agno y otros frameworks

#### 🎯 **SESIÓN 2 - 16 de septiembre** ← **ESTE REPOSITORIO**
- **[BLOQUE 1]** Uso de APIs con llama-index
- **[BLOQUE 1]** Agentes con llama-index
- **[BLOQUE 2]** Refresco de conocimiento sobre Pydantic
- **[BLOQUE 2]** Parseo de documentos usando llama-index, APIs y Pydantic

#### 📌 SESIÓN 3 - 17 de septiembre
- **[BLOQUE 1]** N8N y automatización
- **[BLOQUE 1]** N8N + llama-index: cadena de agentes y "modelo MCP/A2A"
- **[BLOQUE 2]** Agente con @tools automatizando tareas (lectura de correo, navegar una web...)

#### 📌 SESIÓN 4 - 18 de septiembre
- **[BLOQUE 1 y 2]** Mejorando el agente con más @tools y más capacidades. Ampliando el parseo con Pydantic

---

## 🗂️ Estructura del Repositorio

```
curso-llamaindex-pydantic/
├── 📓 notebooks/                    # Jupyter notebooks del curso
│   ├── 01_uso_apis_llamaindex.ipynb
│   ├── 02_agentes_llamaindex.ipynb
│   ├── 03_refresco_pydantic.ipynb
│   ├── 04_parseo_documentos.ipynb
│   ├── extra_retos.ipynb
│   └── sources/                     # Recursos visuales
├── 🏋️ exercises/                    # Ejercicios prácticos
│   ├── bloque1_apis/               # Ejercicios de APIs
│   ├── bloque1_agentes/            # Ejercicios de agentes
│   ├── bloque2_pydantic/           # Ejercicios de Pydantic
│   └── bloque2_parseo/             # Ejercicios de parseo
├── ✅ solutions/                    # Soluciones de los ejercicios
├── 🎯 final_proyect/               # Proyecto final Pokémon
├── 🔧 src/                         # Código fuente del curso
│   ├── apis/                       # APIs de noticias y clima
│   ├── embeddings/                 # Modelos de embeddings
│   └── parsers/                    # Parsers HTML y PDF
├── 📊 data/                        # Datos de ejemplo
├── 🗄️ chroma_db/                   # Base de datos vectorial
├── requirements.txt                # Dependencias Python
└── setup_instructions.md          # Guía de configuración
```

---

## 🎯 Ejercicios Prácticos

### 🔧 Bloque 1: APIs y Agentes

#### 📰 **Ejercicio APIs**: Agente de Noticias con LlamaIndex
- **Objetivo**: Implementar un agente de IA que busque noticias en tiempo real
- **Tecnologías**: NewsAPI, LlamaIndex, Agentes conversacionales
- **Funcionalidades**: Configuración LLM, integración de herramientas, memoria conversacional

#### 🤖 **Ejercicio Agentes**: Sistema Multi-Agente con APIs
- **Objetivo**: Sistema multi-agente con arquitectura orientada a objetos
- **Tecnologías**: NewsAPI, WeatherAPI, ChromaDB, Deduplicación LLM
- **Funcionalidades**: NewsCache, DeduplicationService, AgentThoughtManager

### 🔧 Bloque 2: Pydantic y Parseo

#### ✅ **Ejercicio Pydantic**: Validación Avanzada de Datos Sintéticos
- **Objetivo**: Sistema de validación y generación de datos sintéticos
- **Tecnologías**: Pydantic v2, Validaciones personalizadas, Control de LLMs
- **Funcionalidades**: EnhancedCustomer, AdvancedTransaction, LLMDataValidator

#### 📄 **Ejercicio Parseo**: Análisis y Extracción Documental
- **Objetivo**: Parseo de documentos con agentes especializados
- **Tecnologías**: PDF parsing, HTML parsing, Pydantic models
- **Funcionalidades**: Extracción estructurada, validación de contenido

---

## 🎮 Proyecto Final

*Información del proyecto final será añadida próximamente.*

---

## 🛠️ Configuración del Entorno

### 📋 Requisitos Previos
- **Python 3.9+**
- **Entorno virtual** (recomendado)
- **APIs keys** para servicios externos

### 🔑 Variables de Entorno Requeridas

```bash
# Hugging Face Token (OBLIGATORIO)
HF_TOKEN=hf_tu_token_aqui

# APIs opcionales para ejercicios específicos
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
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# Instalar dependencias
pip install -r requirements.txt

# Configurar variables de entorno
cp .env.example .env
# Editar .env con tus API keys
```

## 📞 Contacto

### 👩‍💼 Coordinadora de Voluntariado
**Rocío Lago**
- 📧 **Email**: [rlago@goodjob.es](mailto:rlago@goodjob.es)
- 📱 **Teléfono**: 605 628 531

### 🏢 Dirección
**Sector Oficios 32, 28760 Tres Cantos**

### 🌐 Web Oficial
[www.fundaciongoodjob.org](https://www.fundaciongoodjob.org)

![Contacto GoodJob](https://via.placeholder.com/600x150/607D8B/FFFFFF?text=Fundaci%C3%B3n+GoodJob+%7C+Tres+Cantos%2C+Madrid)

---

## 🚀 Cómo Empezar

1. **📖 Lee la guía de configuración**: `setup_instructions.md`
2. **🔧 Configura tu entorno** con las APIs necesarias
3. **📓 Explora los notebooks** en orden secuencial
4. **🏋️ Realiza los ejercicios** de cada bloque
5. **✅ Revisa las soluciones** para comparar enfoques

---

## 🤝 Contribuir como Voluntario

Si estás interesado en participar como **voluntario formador** en el programa #IMPACT:

1. **📧 Contacta** con Rocío Lago para más información
2. **📅 Confirma** tu disponibilidad para las fechas del programa
3. **📚 Prepara** tu sesión con los contenidos asignados
4. **🎯 Comparte** tu experiencia profesional con los participantes

---

*Juntos construimos un futuro más inclusivo a través de la tecnología y la formación especializada.*

![Impacto Social](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRGCrL-V9c0NgujafmRoO5ZIL-92l0GF5LvMA&s)

---

**© 2025 Fundación GoodJob - Programa de Voluntariado Corporativo #IMPACT**