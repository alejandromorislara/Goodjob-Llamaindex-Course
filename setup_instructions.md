# 🚀 Configuración del Curso - LlamaIndex & Pydantic

¡Bienvenid@ al curso de **IA Generativa** de la [Fundación GoodJob](https://www.fundaciongoodjob.org/)! 🎉

Este documento te guiará paso a paso para configurar todo lo necesario y empezar a crear agentes inteligentes. No te preocupes si eres principiante, ¡lo haremos juntos de forma sencilla!

## 📋 ¿Qué vamos a configurar?

Durante el curso trabajaremos con:
- 🤖 **Modelos LLM** gratuitos via Hugging Face
- 🌤️ **API del clima** (OpenWeatherMap) 
- 📰 **API de noticias** (NewsAPI)
- 🐍 **Python** y librerías especializadas

---

## 🐍 Paso 1: Instalación de Python

### Windows
1. Ve a [python.org/downloads](https://www.python.org/downloads/)
2. Descarga **Python 3.9 o superior**
3. Durante la instalación, **marca la casilla "Add Python to PATH"** ✅
4. Verifica la instalación abriendo cmd y escribiendo: `python --version`

### macOS
```bash
# Con Homebrew (recomendado)
brew install python@3.9

# O descarga desde python.org
```

### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install python3.9 python3.9-pip python3.9-venv
```

---

## 📦 Paso 2: Configuración del Proyecto

### 1. Clona o descarga el repositorio
```bash
# Si tienes git
git clone [URL_DEL_REPOSITORIO]
cd curso-llamaindex-pydantic

# O descarga el ZIP y extráelo
```

### 2. Crea un entorno virtual (¡Muy importante!)
```bash
# En Windows
python -m venv .venv
.venv\Scripts\activate

# En macOS/Linux  
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Instala las dependencias
```bash
pip install -r requirements.txt
```

---

## 🔑 Paso 3: Tokens y APIs 

### 🤗 Hugging Face Token (OBLIGATORIO)

**¿Por qué lo necesitas?** Para usar modelos de IA gratuitos en la nube.

**Cómo conseguirlo:**
1. Ve a [huggingface.co](https://huggingface.co) y **crea una cuenta gratuita**
2. Verifica tu correo electrónico
2. Una vez logueado, ve a tu perfil → **Settings** → **Access Tokens**
3. Haz clic en **"Create new token"**
4. Ponle un nombre como `"curso-goodjob"` 
5. Selecciona tipo **"Read"** (suficiente para el curso)
6. Copia el token que empieza por `hf_...`

**Límites gratuitos:** 1000 requests/día. ¡Más que suficiente para el curso!

### 🌤️ OpenWeatherMap API (Para el agente del clima)

**¿Para qué?** Tu agente podrá consultar el tiempo actual de cualquier ciudad.

**Cómo conseguirlo:**
1. Ve a [openweathermap.org](https://openweathermap.org)
2. Haz clic en **"Sign Up"** y crea una cuenta gratuita
3. Confirma tu email
4. Ve a **"API Keys"** en tu panel de usuario
5. Copia tu **Default API key**

**Límites gratuitos:** 1000 llamadas/día, 60 llamadas/minuto.

### 📰 NewsAPI (Para el agente de noticias)

**¿Para qué?** Tu agente podrá buscar noticias actuales sobre cualquier tema.

**Cómo conseguirlo:**
1. Ve a [newsapi.org](https://newsapi.org)
2. Haz clic en **"Get API Key"**
3. Completa el registro (selecciona **"Developer"** si es para uso personal)
4. Confirma tu email
5. Copia tu **API Key** desde el dashboard

**Límites gratuitos:** 1000 requests/día.


---

## 🗂️ Paso 4: Configurar Variables de Entorno

### Opción A: Archivo .env (Recomendado)

Crea un archivo llamado `.env` en la carpeta raíz del proyecto con:

```env
# Hugging Face Token (OBLIGATORIO)
HF_TOKEN=hf_tu_token_aqui

# APIs opcionales (para agentes específicos)
OPENWEATHER_KEY=tu_api_key_de_openweather
NEWS_API_KEY=tu_api_key_de_newsapi


### Opción B: Variables de Sistema

#### Windows
```cmd
setx HF_TOKEN "hf_tu_token_aqui"
setx OPENWEATHER_KEY "tu_api_key"
setx NEWS_API_KEY "tu_api_key"
```

#### macOS/Linux
```bash
export HF_TOKEN="hf_tu_token_aqui"
export OPENWEATHER_KEY="tu_api_key"
export NEWS_API_KEY="tu_api_key"
```

---

## ✅ Paso 5: Verificar la Configuración

### Test rápido de Hugging Face
```bash
python -c "
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
import os
from dotenv import load_dotenv

load_dotenv()
llm = HuggingFaceInferenceAPI(
    model_name='HuggingFaceTB/SmolLM3-3B',
    token=os.getenv('HF_TOKEN')
)
print('✅ Hugging Face configurado correctamente!')
"
```

### Test de las APIs (opcional)
```bash
# Clima
python src/apis/weather_api.py --city "Madrid"

# Noticias  
python src/apis/news_api.py --query "tecnología" --page_size 3
```

---

## 🚨 Resolución de Problemas

### ❌ "ModuleNotFoundError"
**Problema:** No activaste el entorno virtual o no instalaste las dependencias.
**Solución:**
```bash
# Activa el entorno virtual
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # macOS/Linux

# Reinstala dependencias
pip install -r requirements.txt
```

### ❌ "Token inválido" en Hugging Face
**Problema:** El token está mal copiado o caducado.
**Solución:**
1. Ve a [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Crea un nuevo token
3. Actualiza tu archivo `.env`

### ❌ "API Key inválida" en OpenWeather/NewsAPI
**Problema:** La clave está mal copiada o no está activada.
**Solución:**
1. Revisa que copiaste la clave completa
2. Para OpenWeather, espera unos minutos tras crear la cuenta (a veces tarda en activarse)
3. Para NewsAPI, confirma tu email


### ❌ "Permission denied" al crear entorno virtual
**Problema:** Permisos en Windows.
**Solución:**
```bash
# Ejecuta PowerShell como administrador
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

## 📚 ¿Qué sigue?

Una vez configurado todo:

1. 🎯 **Empieza con el notebook:** `notebooks/01_uso_apis_llamaindex.ipynb`
2. 🤖 **Prueba los agentes:** `agente_avanzado.py`
3. 🏗️ **Haz los ejercicios:** Carpeta `exercises/`

---

## 🆘 ¿Necesitas Ayuda?

- 📧 **Email:** [Contacto Instructor Fundación GoodJob](mailto:alejandrgi2g@gmail.com)
- 🐛 **Problemas técnicos:** Revisa la sección de resolución de problemas arriba

---

> [!tip]  
> **💡 Consejos Extras**  
>   
> **Para ahorrar requests:**  
> - Las APIs tienen límites diarios, úsalas con moderación durante las pruebas  
> - Puedes usar datos de ejemplo si agotas tu cuota  
>   
> **Seguridad:**  
> - ⚠️ **NUNCA subas tu archivo `.env` a GitHub**  
> - Añade `.env` a tu `.gitignore`  
> - Si accidentalmente subes tokens, créalos de nuevo  

---

¡**Ya estás listo para crear agentes inteligentes!** 🎉
