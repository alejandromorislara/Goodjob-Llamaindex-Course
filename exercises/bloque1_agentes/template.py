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
    
    TODO: Implementa una de estas opciones:
    - OpenAI (requiere OPENAI_API_KEY)
    - HuggingFace (requiere HUGGINGFACE_API_KEY o usa gratis)
    
    Returns:
        LLM configurado y listo para usar
    """
    # TODO: Implementar configuración del LLM
    # Pista: Revisa el notebook para ver ejemplos de configuración
    
    # Opción A: OpenAI (recomendado si tienes API key)
    # llm = OpenAI(
    #     model="gpt-4o-mini",
    #     api_key=os.getenv("OPENAI_API_KEY")
    # )
    
    # Opción B: HuggingFace (gratuito)
    # llm = HuggingFaceInferenceAPI(
    #     model_name="HuggingFaceTB/SmolLM3-3B",
    #     max_new_tokens=1024,
    #     temperature=0.1
    # )
    
    pass  # Reemplaza esto con tu implementación


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
    # TODO: Importar la herramienta
    # from src.apis.news_api import news_search_tool
    
    # TODO: Crear la herramienta
    # news_tool = news_search_tool()
    
    # TODO: Crear el agente
    # agent = AgentWorkflow.from_tools_or_functions(
    #     tools_or_functions=[news_tool],
    #     llm=llm,
    #     system_prompt=(
    #         "Eres un asistente especializado en búsqueda de noticias. "
    #         "Puedes buscar noticias actualizadas usando tu herramienta. "
    #         "Responde en español de manera amigable y profesional. "
    #         "Recuerda las preferencias del usuario durante la conversación."
    #     ),
    # )
    
    pass  # Reemplaza esto con tu implementación


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
    # TODO: Implementar creación de contexto
    # Pista: Context(agent)
    
    pass  # Reemplaza esto con tu implementación


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
        
        # TODO: Mostrar el proceso de herramientas
        # async for ev in handler.stream_events():
        #     if isinstance(ev, ToolCallResult):
        #         print(f"\n🔧 Buscando noticias...")
        #         print(f"📡 Parámetros: {ev.tool_kwargs}")
        #         print(f"📊 Encontradas: {len(ev.tool_output.articles) if hasattr(ev.tool_output, 'articles') else 'N/A'}")
        #         print("🤖 Agente: ", end="", flush=True)
        #     elif isinstance(ev, AgentStream):
        #         print(ev.delta, end="", flush=True)
        
        # TODO: Obtener respuesta final
        # response = await handler
        
        # TODO: Extraer contenido de la respuesta
        # response_text = response.response.content
        # print(response_text)
        
        print("\n" + "="*60)
        # return response
        
        pass  # Reemplaza esto con tu implementación
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("🔄 Verifica tu configuración de API keys y conexión a internet.")
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
    
    required_packages = ['llama_index', 'openai', 'requests', 'pydantic', 'python-dotenv']
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - Instala con: pip install {package}")
    
    print("\n🚀 Iniciando ejercicio...")
    
    # Ejecutar en entorno async
    import asyncio
    asyncio.run(main())


# =============================================================================
# NOTAS PARA EL ESTUDIANTE
# =============================================================================

"""
📝 NOTAS IMPORTANTES:

1. VARIABLES DE ENTORNO:
   - Crea un archivo .env en la raíz del proyecto
   - Añade: NEWS_API_KEY=tu_clave_de_newsapi
   - Opcional: OPENAI_API_KEY=tu_clave (si usas OpenAI)

2. APIS GRATUITAS:
   - NewsAPI: https://newsapi.org/ (500 requests/día gratis)
   - HuggingFace: https://huggingface.co/ (uso gratuito limitado)

3. DEBUGGING:
   - Si hay errores de Pydantic, verifica que news_api.py esté actualizado
   - Si falla la importación, verifica el path de src/
   - Si no encuentra noticias, verifica tu API key

4. MEJORAS EXTRAS (+1 punto):
   - Añade filtros por fecha
   - Formatea respuestas con emojis
   - Implementa resúmenes de noticias
   - Añade soporte multiidioma

¡Buena suerte! 🍀
"""