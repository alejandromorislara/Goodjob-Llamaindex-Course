"""
🤖 Template: Agente de Noticias con LlamaIndex

INSTRUCCIONES:
1. Completa las funciones marcadas con # TODO
2. Configura las variables de entorno en .env
3. Ejecuta y prueba las conversaciones de ejemplo

ESTUDIANTE: ___________________
FECHA: _______________________
"""

import os
import sys
from typing import Optional
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Imports necesarios para LlamaIndex
from llama_index.core.workflow import Context
from llama_index.core.agent.workflow import AgentWorkflow, ToolCallResult, AgentStream
from llama_index.llms.openai import OpenAI
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI

# Agregar src al path para importar news_api
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))


# =============================================================================
# PARTE 1: CONFIGURACIÓN DEL LLM (2.5 puntos)
# =============================================================================

def setup_llm():
    """
    Configura y retorna un modelo de lenguaje.
    
    TODO: Implementa estas opciones:
    - OpenAI (requiere OPENAI_API_KEY)
    - HuggingFace (requiere HUGGINGFACE_API_KEY)
    
    Returns:
        LLM configurado y listo para usar
    """
    # Obtener las claves de API del entorno
    openai_key = os.getenv("___________")  # TODO: ¿Qué variable necesitas?
    hf_key = os.getenv("HUGGINGFACE_API_KEY")
    
    # Opción A: Intenta OpenAI primero si está disponible
    if openai_key:
        try:
            llm = OpenAI(
                model="gpt-4o-mini",
                api_key=openai_key,
                temperature=___  # TODO: ¿Qué temperatura usar? (0.0-1.0)
            )
            print("✅ Usando OpenAI GPT-4o-mini")
            return ___  # TODO: ¿Qué retornar?
        except Exception as e:
            print(f"⚠️ Error configurando OpenAI: {e}")
    
    # TODO: Opción B: Fallback a HuggingFace
    pass


# =============================================================================
# PARTE 2: INTEGRACIÓN DE LA HERRAMIENTA (2.5 puntos)
# =============================================================================

def create_news_agent(llm):
    """
    Crea un agente que puede buscar noticias usando news_search_tool.
    
    TODO: 
    1. Importar news_search_tool de src.apis.news_api
    2. Crear un AgentWorkflow con la herramienta
    3. Configurar un system_prompt apropiado
    
    Args:
        llm: Modelo de lenguaje configurado
        
    Returns:
        AgentWorkflow configurado con herramienta de noticias
    """
    try:
        # TODO: Importar la herramienta de noticias
        from src.apis.news_api import ___________  # ¿Qué función importar?
        
        
        news_tool = news_search_tool() 
        
        # TODO: Crear el agente con las herramientas
        pass
        
    except ImportError as e:
        print(f"❌ Error creando agente: {e}")
        return None


# =============================================================================
# PARTE 3: MEMORIA CONVERSACIONAL (2.5 puntos)
# =============================================================================

def create_conversation_context(agent):
    """
    Crea un contexto de conversación para mantener memoria.
    
    TODO: Crear y retornar un Context para el agente
    
    Args:
        agent: AgentWorkflow configurado
        
    Returns:
        Context para mantener memoria conversacional
    """
    try:
        pass
        
    except Exception as e:
        print(f"❌ Error creando contexto: {e}")
        return None


# =============================================================================
# PARTE 4: CONVERSACIÓN INTELIGENTE (2.5 puntos)
# =============================================================================

async def chat_with_news_agent(message: str, agent, context):
    """
    Maneja una conversación con el agente de noticias mostrando el proceso.
    
    TODO:
    1. Ejecutar agent.run() con el mensaje y contexto
    2. Mostrar el proceso usando handler.stream_events()
    3. Manejar errores graciosamente
    4. Retornar la respuesta final
    
    Args:
        message: Mensaje del usuario
        agent: AgentWorkflow configurado
        context: Context para memoria
        
    Returns:
        Respuesta del agente
    """
    print(f"👤 Usuario: {message}")
    print("🤖 Agente: ", end="", flush=True)
    
    try:
        # TODO: Ejecutar el agente
        # handler = agent.run(message, ctx=context)
        
        response_text = ""
        tool_calls_made = 0
        
        # TODO: Procesar eventos en streaming
        async for ev in handler.stream_events():
            if isinstance(ev, ___):  # TODO: ¿Qué clase es?
                tool_calls_made += 1
                print(f"\n🔧 Usando herramienta de búsqueda...")
                
                # TODO: Mostrar parámetros de búsqueda
                if hasattr(ev, 'tool_kwargs') and ev.tool_kwargs:
                    params = ev.tool_kwargs
                    print(f"📡 Parámetros: query='{params.get('___', 'N/A')}'")
                
                # TODO: Mostrar número de resultados
                if hasattr(ev, 'tool_output'):
                    output = ev.tool_output
                    if hasattr(output, 'articles'):
                        article_count = len(output._____)  # ¿Qué atributo?
                        print(f"📊 Encontradas: {article_count} noticias")
                
                print("🤖 Procesando resultados: ", end="", flush=True)
                
            elif isinstance(ev, AgentStream):
                # TODO: Mostrar el texto de respuesta en streaming
                print(ev.___, end="", flush=True)  # ¿Qué atributo contiene el texto?
                response_text += ev.delta
        
        # TODO: Obtener respuesta final
        response = await ___  # ¿Qué esperar?
        
        print(f"\n{'='*60}")
        
        # Mostrar resumen
        if tool_calls_made > 0:
            print(f"💫 Resumen: Se utilizaron {tool_calls_made} herramienta(s)")
        
        return ___  # TODO: ¿Qué retornar?
        
    except Exception as e:
        print(f"❌ Error durante la conversación: {e}")
        print("🔄 Verifica tu configuración de API keys y conexión a internet.")
        print(f"📝 Detalles del error: {type(e).__name__}")
        return None


# =============================================================================
# FUNCIÓN PRINCIPAL Y CASOS DE PRUEBA
# =============================================================================

async def main():
    """
    Función principal que prueba todas las funcionalidades.
    """
    print("🚀 Iniciando Agente de Noticias con LlamaIndex")
    print("="*60)
    
    # Verificar variables de entorno
    if not os.getenv("NEWS_API_KEY"):
        print("❌ Error: NEWS_API_KEY no configurada")
        print("💡 Obtén tu clave gratuita en: https://newsapi.org/")
        return
    
    try:
        # PARTE 1: Configurar LLM
        print("\n1️⃣ Configurando modelo de lenguaje...")
        llm = setup_llm()
        if llm is None:
            print("❌ Error: No se pudo configurar el LLM")
            return
        print("✅ LLM configurado correctamente")
        
        # PARTE 2: Crear agente
        print("\n2️⃣ Creando agente de noticias...")
        agent = create_news_agent(llm)
        if agent is None:
            print("❌ Error: No se pudo crear el agente")
            return
        print("✅ Agente creado correctamente")
        
        # PARTE 3: Crear contexto
        print("\n3️⃣ Configurando memoria conversacional...")
        context = create_conversation_context(agent)
        if context is None:
            print("❌ Error: No se pudo crear el contexto")
            return
        print("✅ Memoria configurada correctamente")
        
        # PARTE 4: Casos de prueba
        print("\n4️⃣ Ejecutando casos de prueba...")
        
        test_cases = [
            "Busca noticias sobre inteligencia artificial",
            "Mi nombre es Carlos. Busca noticias sobre OpenAI en español", 
            "¿Recuerdas mi nombre? Busca más noticias sobre IA",
            "Busca 5 noticias recientes sobre tecnología ordenadas por relevancia"
        ]
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n🧪 Test {i}/{len(test_cases)}:")
            response = await chat_with_news_agent(test_case, agent, context)
            
            if response is None:
                print(f"❌ Test {i} falló")
                break
            
            print(f"✅ Test {i} completado")
        
        print(f"\n🎉 ¡Ejercicio completado! Revisa los resultados arriba.")
        
    except Exception as e:
        print(f"❌ Error general: {e}")
        print("🔍 Revisa tu implementación y configuración")


# =============================================================================
# EJECUCIÓN
# =============================================================================

if __name__ == "__main__":
    # Verificación rápida del entorno
    print("🔍 Verificando entorno...")
    
    required_packages = ['llama_index', 'openai', 'requests', 'pydantic', 'load_dotenv']
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - Instala con: pip install {package}")
    
    print("\n🚀 Iniciando ejercicio...")
    
    # Ejecutar en entorno async (asyncio.run(main()) De normal en un jupyter notebook ya hay un proceso asíncrono, luego no hace falta)
    import asyncio
    asyncio.run(main())


# =============================================================================
# NOTAS PARA EL ESTUDIANTE
# =============================================================================

"""
📝 PISTAS Y AYUDAS PARA COMPLETAR EL EJERCICIO:

🔧 PARTE 1 - CONFIGURACIÓN LLM:
   - Variable de entorno OpenAI: "OPENAI_API_KEY"
   - Temperatura recomendada: 0.1 (más determinista)
   - Si OpenAI falla, devolver None y intentar HuggingFace

🛠️ PARTE 2 - INTEGRACIÓN HERRAMIENTA:
   - Función a importar: "news_search_tool"
   - Lista de herramientas: [news_tool]
   - Verbose=True para ver el proceso
   - Si falla importación, devolver None

🧠 PARTE 3 - MEMORIA CONVERSACIONAL:
   - Context necesita el agente como parámetro
   - Retornar el context creado

💬 PARTE 4 - CONVERSACIÓN INTELIGENTE:
   - handler = agent.run(message, ctx=context)
   - Parámetro de búsqueda: 'query'
   - Atributo de artículos: 'articles'
   - Texto del stream: ev.delta
   - Esperar: handler
   - Retornar: response

🌐 VARIABLES DE ENTORNO NECESARIAS:
   - NEWS_API_KEY=tu_clave_de_newsapi (OBLIGATORIO)
   - OPENAI_API_KEY=tu_clave_openai (OPCIONAL)
   - HUGGINGFACE_API_KEY=tu_clave_hf (OBLIGATORIO)

🔗 APIS GRATUITAS:
   - NewsAPI: https://newsapi.org/ (500 requests/día)
   - HuggingFace: https://huggingface.co/ (uso limitado)

🐛 DEBUGGING COMÚN:
   - ImportError: Ejecuta desde la raíz del proyecto
   - Error de API: Verifica tu NEWS_API_KEY
   - Error de LLM: Configura al menos una API key

🌟 MEJORAS EXTRAS (+1 punto):
   - Validación de entorno automática
   - Formateo de respuestas con emojis  
   - Manejo de errores más robusto
   - Resúmenes de interacción

¡Completa los ___ y TODO para que funcione! 🚀
"""