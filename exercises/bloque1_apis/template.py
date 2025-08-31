"""
🌐 Template: Sistema Multi-Agente con APIs y ChromaDB

INSTRUCCIONES:
1. Completa las funciones marcadas con # TODO
2. Configura las variables de entorno en .env
3. Ejecuta y prueba las consultas de ejemplo

ESTUDIANTE: ___________________
FECHA: _______________________
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Imports necesarios
from llama_index.core import Settings, Document, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.tools import FunctionTool, QueryEngineTool
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

# =============================================================================
# PARTE 1: CONFIGURACIÓN Y CHROMADB (2.5 puntos)
# =============================================================================

def setup_environment():
    """TODO: Configurar LLM y embeddings"""
    openai_key = os.getenv("___________")  # ¿Qué variable?
    
    if openai_key:
        Settings.llm = OpenAI(
            model="gpt-4o-mini",
            api_key=openai_key,
            temperature=___  # ¿Qué temperatura?
        )
    
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="___________"  # ¿Qué modelo?
    )
    return True

def create_vector_database():
    """TODO: Crear ChromaDB con documentos"""
    sample_docs = [
        Document(text="""POLÍTICA DE TRANSFERENCIAS
        SWIFT UE: 15 EUR + 0.2%
        SWIFT USA: 25 USD + 0.3%
        SEPA >1000 EUR: gratis"""),
        # TODO: Añadir más documentos
    ]
    
    parser = SentenceSplitter(chunk_size=___, chunk_overlap=___)
    nodes = parser.get_nodes_from_documents(sample_docs)
    
    client = chromadb.PersistentClient(path="../../chroma_db")
    collection = client.get_or_create_collection("___________")
    
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    index_docs = VectorStoreIndex(___, storage_context=storage_context)
    qe_docs = index_docs.as_query_engine(similarity_top_k=___)
    
    return qe_docs

# =============================================================================
# PARTE 2: HERRAMIENTAS (2.5 puntos)
# =============================================================================

def create_api_tools():
    """TODO: Crear herramientas de API"""
    def fx_lookup(base: str, quote: str) -> dict:
        mock_rates = {("EUR", "USD"): 1.08, ("USD", "EUR"): 0.93}
        rate = mock_rates.get((base.upper(), quote.upper()))
        return {"rate": rate} if rate else {"error": "No soportado"}
    
    def current_weather(city: str) -> dict:
        # TODO: Implementar clima mock
        pass
    
    fx_tool = FunctionTool.from_defaults(
        fn=fx_lookup, 
        name="___________",
        description="___________"
    )
    
    weather_tool = FunctionTool.from_defaults(
        fn=current_weather, 
        name="current_weather",
        description="Consulta clima de una ciudad"
    )
    
    return [fx_tool, weather_tool]

def create_math_tools():
    """TODO: Crear calculadora segura"""
    def safe_calculator(expression: str) -> dict:
        import re
        if not re.fullmatch(r"___________", expression):
            return {"error": "Expresión no permitida"}
        
        try:
            result = eval(expression)
            return {"result": result}
        except Exception as e:
            return {"error": str(e)}
    
    calc_tool = FunctionTool.from_defaults(
        fn=safe_calculator, 
        name="calculator",
        description="___________"
    )
    
    return [calc_tool]

# =============================================================================
# PARTE 3: AGENTES Y ROUTER (2.5 puntos)
# =============================================================================

def create_agents(qe_docs, api_tools, math_tools):
    """TODO: Crear agentes especializados"""
    policy_tool = QueryEngineTool.from_defaults(
        query_engine=qe_docs,
        name="___________",
        description="___________"
    )
    
    docs_agent = ReActAgent.from_tools([policy_tool], llm=Settings.llm, max_iterations=___)
    api_agent = ReActAgent.from_tools(api_tools, llm=Settings.llm, max_iterations=4)
    math_agent = ReActAgent.from_tools(math_tools, llm=Settings.llm, max_iterations=___)
    
    return {"docs": docs_agent, "api": api_agent, "math": math_agent}

def create_router():
    """TODO: Crear router inteligente"""
    def route_with_rules(message: str) -> str:
        msg_lower = message.lower()
        docs_keywords = ["___________"]  # ¿Qué keywords?
        api_keywords = ["___________"]
        math_keywords = ["___________"]
        
        # TODO: Implementar lógica de routing
        return "unknown"
    
    def smart_router(message: str) -> str:
        # TODO: Combinar reglas + LLM
        pass
    
    return smart_router

# =============================================================================
# PARTE 4: ORQUESTACIÓN (2.5 puntos)
# =============================================================================

def create_orchestrator(agents, router):
    """TODO: Crear orquestador multi-agente"""
    def smart_chat(message: str):
        # TODO: Detectar consultas complejas
        # TODO: Coordinar múltiples agentes
        # TODO: Combinar resultados
        pass
    
    return smart_chat

# =============================================================================
# FUNCIÓN PRINCIPAL
# =============================================================================

def main():
    print("🚀 Sistema Multi-Agente con ChromaDB")
    
    # TODO: Ejecutar todas las partes
    setup_environment()
    qe_docs = create_vector_database()
    api_tools = create_api_tools()
    math_tools = create_math_tools()
    agents = create_agents(qe_docs, api_tools, math_tools)
    router = create_router()
    smart_chat = create_orchestrator(agents, router)
    
    # Casos de prueba
    test_cases = [
        "¿Comisión SWIFT a USA?",
        "¿Cambio EUR/USD?",
        "Calcula 15% de 2500",
        "Si envío 1200 EUR por SWIFT a USA, ¿cuánto llega?"
    ]
    
    for test in test_cases:
        print(f"\n🧪 Test: {test}")
        # TODO: Ejecutar smart_chat(test)

if __name__ == "__main__":
    main()

# =============================================================================
# PISTAS
# =============================================================================
"""
🔧 CONFIGURACIÓN:
- Variable OpenAI: "OPENAI_API_KEY"
- Temperatura: 0.1
- Embeddings: "sentence-transformers/all-MiniLM-L6-v2"
- Chunk size: 500, overlap: 50
- Colección: "company_policies"

🛠️ HERRAMIENTAS:
- Regex calculadora: r"[0-9\.\+\-\*\/\(\) ]+"
- Nombres: "fx_lookup", "current_weather", "calculator"
- Herramienta docs: "policies_search"

🎯 ROUTER:
- Keywords docs: ["documento", "política", "comisión"]
- Keywords api: ["cambio", "clima", "eur", "usd"]
- Keywords math: ["calcula", "suma", "%"]

🎼 ORQUESTACIÓN:
- Detectar: "swift" + divisas
- Secuencia: Docs → Math → API
"""
