"""
🌐 Template: Sistema Multi-Agente con APIs de Noticias y Clima - Arquitectura con Clases

INSTRUCCIONES:
1. Completa las clases marcadas con # TODO
2. Configura las variables de entorno en .env
3. Ejecuta y prueba las consultas de ejemplo

ESTUDIANTE: ___________________
FECHA: _______________________
"""

import os
import sys
import json
import asyncio
from datetime import datetime
from enum import Enum
from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Agregar src al path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Cargar variables de entorno
load_dotenv()

# Imports de LlamaIndex
from llama_index.core import Settings, Document, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.tools import FunctionTool
from llama_index.core.agent.workflow import FunctionAgent, AgentWorkflow
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

# Imports externos
import chromadb
from chromadb.config import Settings as ChromaSettings

# Imports de APIs
from src.apis.news_api import fetch_news
from src.apis.weather_api import fetch_current_weather

# =============================================================================
# MODELOS PYDANTIC
# =============================================================================

class DeduplicationDecision(str, Enum):
    """TODO: Enum para decisión de deduplicación"""
    INSERT = "___________"  # TODO: ¿Qué valor para insertar?
    SKIP = "___________"    # TODO: ¿Qué valor para omitir?

class ArticleComparisonResult(BaseModel):
    """TODO: Modelo para resultado de comparación individual de artículo"""
    decision: DeduplicationDecision = Field(description="___________")  # TODO: ¿Descripción?
    reasoning: str = Field(description="___________")  # TODO: ¿Descripción del razonamiento?
    confidence: float = Field(description="___________", ge=0, le=1)  # TODO: ¿Descripción?
    article_title: str = Field(description="___________")  # TODO: ¿Descripción?

class DeduplicationResult(BaseModel):
    """TODO: Modelo para resultado completo de deduplicación"""
    total_articles: int = Field(description="___________")  # TODO: ¿Descripción?
    articles_inserted: int = Field(description="___________")  # TODO: ¿Descripción?
    articles_skipped: int = Field(description="___________")  # TODO: ¿Descripción?
    individual_results: List[ArticleComparisonResult] = Field(description="___________")  # TODO: ¿Descripción?

class AgentThought(BaseModel):
    """TODO: Modelo para pensamiento del agente"""
    agent_name: str = Field(description="___________")  # TODO: ¿Descripción?
    timestamp: str = Field(description="___________")  # TODO: ¿Descripción?
    query: str = Field(description="___________")  # TODO: ¿Descripción?
    reasoning: str = Field(description="___________")  # TODO: ¿Descripción?
    decision: str = Field(description="___________")  # TODO: ¿Descripción?
    confidence: float = Field(description="___________")  # TODO: ¿Descripción?

# =============================================================================
# CLASE 1: CONFIGURACIÓN DEL SISTEMA (1 punto)
# =============================================================================

class SystemConfig:
    """TODO: Configuración global del sistema"""
    
    @staticmethod
    def verify_environment() -> bool:
        """TODO: Verificar que todas las dependencias estén instaladas"""
        print("🔍 Verificando entorno...")
        
        required_packages = [
            "___________", "___________", "___________", "___________", "___________"  # TODO: ¿Qué paquetes verificar?
        ]
        
        missing = []
        for package in required_packages:
            try:
                __import__(package.replace("-", "_"))
                print(f"✅ {package}")
            except ImportError:
                missing.append(package)
                print(f"❌ {package}")
        
        if missing:
            print(f"\n❌ Faltan paquetes: {', '.join(missing)}")
            print("Ejecuta: pip install " + " ".join(missing))
            return False
        
        return True

    @staticmethod
    def setup_llm_environment():
        """TODO: Configurar el entorno LlamaIndex"""
        # TODO: Configurar LLM
        Settings.llm = OpenAI(
            model="___________",  # TODO: ¿Qué modelo usar?
            temperature=___,  # TODO: ¿Qué temperatura?
            api_key=os.getenv("___________")  # TODO: ¿Qué variable de entorno?
        )
        
        # TODO: Configurar embeddings
        Settings.embed_model = OpenAIEmbedding(
            model="___________",  # TODO: ¿Qué modelo de embeddings?
            api_key=os.getenv("___________")  # TODO: ¿Qué variable de entorno?
        )
        
        # TODO: Configurar text splitter
        Settings.text_splitter = SentenceSplitter(chunk_size=___, chunk_overlap=___)  # TODO: ¿Qué valores?
        
        print("✅ Usando OpenAI GPT-4o-mini")

# =============================================================================
# CLASE 2: GESTIÓN DE PENSAMIENTOS DE AGENTES (1 punto)
# =============================================================================

class AgentThoughtManager:
    """TODO: Gestor de pensamientos de agentes"""
    
    def __init__(self):
        self.thoughts: List[AgentThought] = []
    
    def save_thought(self, agent_name: str, query: str, reasoning: str, decision: str, confidence: float):
        """TODO: Guardar pensamiento del agente"""
        thought = AgentThought(
            agent_name=agent_name,
            timestamp=datetime.now().isoformat(),
            query=query,
            reasoning=reasoning,
            decision=decision,
            confidence=confidence
        )
        # TODO: Agregar pensamiento a la lista
        self.thoughts.append(thought)
        print(f"💭 {agent_name}: {decision} (confianza: {confidence:.2f})")
    
    def get_recent_thoughts(self, limit: int = 5) -> str:
        """TODO: Obtener pensamientos recientes del agente"""
        print(f"🔧 AgentThoughtManager → get_recent_thoughts(limit={limit})")
        
        if not self.thoughts:
            return "📝 No hay pensamientos guardados aún."
        
        # TODO: Obtener últimos pensamientos
        recent_thoughts = self.thoughts[-limit:]
        result = f"💭 Pensamientos del agente ({len(recent_thoughts)} registros):\n\n"
        for i, thought in enumerate(recent_thoughts, 1):
            result += f"{i}. **{thought.agent_name}** ({thought.timestamp})\n"
            result += f"   Query: {thought.query}\n"
            result += f"   Decisión: {thought.decision}\n"
            result += f"   Razonamiento: {thought.reasoning}\n"
            result += f"   Confianza: {thought.confidence:.2f}\n\n"
        
        print(f"✅ AgentThoughtManager ← get_recent_thoughts: {len(recent_thoughts)} pensamientos")
        return result

# =============================================================================
# CLASE 3: CACHE DE NOTICIAS (2 puntos)
# =============================================================================

class NewsCache:
    """TODO: Cache de noticias con ChromaDB"""
    
    def __init__(self, db_path: str = "___________"):  # TODO: ¿Qué path usar?
        self.db_path = db_path
        self.chroma_client = None
        self.news_collection = None
        self.news_index = None
        
    def initialize(self):
        """TODO: Inicializar cache de noticias con ChromaDB"""
        # TODO: Configurar ChromaDB
        self.chroma_client = chromadb.PersistentClient(
            path=self.db_path,
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        
        # TODO: Obtener o crear colección
        self.news_collection = self.chroma_client.get_or_create_collection(
            name="___________",  # TODO: ¿Nombre de la colección?
            metadata={"description": "Cache de noticias"}
        )
        
        # TODO: Crear vector store
        vector_store = ChromaVectorStore(chroma_collection=self.news_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        # TODO: Crear índice
        self.news_index = VectorStoreIndex([], storage_context=storage_context)
        
        print("✅ Cache de noticias configurado")
    
    def search(self, query: str) -> str:
        """TODO: Buscar en cache de noticias"""
        try:
            # TODO: Crear query engine
            query_engine = self.news_index.as_query_engine(similarity_top_k=___)  # TODO: ¿Cuántos resultados?
            response = query_engine.query(query)
            return response.response if response.response else ""
        except Exception as e:
            print(f"⚠️ Error buscando en cache: {e}")
            return ""
    
    def search_similar_articles_by_title(self, title: str, top_k: int = 3) -> List[Dict]:
        """TODO: Buscar artículos similares por título en la BBDD"""
        if self.news_index is None:
            return []
        
        try:
            # TODO: Buscar por título específico
            retriever = self.news_index.as_retriever(similarity_top_k=top_k)
            nodes = retriever.retrieve(title)
            
            similar_articles = []
            for node in nodes:
                # TODO: Extraer información del nodo
                metadata = node.metadata if hasattr(node, 'metadata') else {}
                similar_articles.append({
                    'title': metadata.get('title', 'Sin título'),
                    'description': metadata.get('description', node.text[:200] + "..." if len(node.text) > 200 else node.text),
                    'source': metadata.get('source', 'Desconocida'),
                    'similarity_score': node.score if hasattr(node, 'score') else 0.0,
                    'content': node.text
                })
            
            return similar_articles
        except Exception as e:
            print(f"⚠️ Error buscando artículos similares: {e}")
            return []
    
    def insert_article(self, article: Dict, query: str):
        """TODO: Insertar un artículo en el cache"""
        # TODO: Crear documento
        doc = Document(
            text=f"{article.get('title', '')}\n\n{article.get('description', '')}",
            metadata={
                "title": article.get('title', ''),
                "description": article.get('description', ''),
                "url": article.get('url', ''),
                "source": article.get('source', ''),
                "published_at": article.get('publishedAt', ''),
                "query": query
            }
        )
        # TODO: Insertar documento en el índice
        self.news_index.insert(doc)

# =============================================================================
# CLASE 4: SERVICIO DE DEDUPLICACIÓN LLM (2.5 puntos)
# =============================================================================

class DeduplicationService:
    """TODO: Servicio de deduplicación usando LLM"""
    
    def __init__(self, news_cache: NewsCache, thought_manager: AgentThoughtManager):
        self.news_cache = news_cache
        self.thought_manager = thought_manager
    
    def compare_article_with_existing(self, new_article: Dict, similar_articles: List[Dict]) -> ArticleComparisonResult:
        """TODO: Comparar un artículo nuevo con artículos similares existentes usando LLM"""
        new_title = new_article.get('title', '')
        new_description = new_article.get('description', '')
        
        if not similar_articles:
            # TODO: No hay artículos similares, insertar
            return ArticleComparisonResult(
                decision=DeduplicationDecision.___________,  # TODO: ¿Qué decisión?
                reasoning="___________",  # TODO: ¿Qué razonamiento?
                confidence=___,  # TODO: ¿Qué confianza?
                article_title=new_title
            )
        
        # TODO: Preparar información de artículos existentes
        existing_info = ""
        for i, art in enumerate(similar_articles[:3], 1):  # Top 3
            existing_info += f"{i}. Título: {art['title']}\n"
            existing_info += f"   Descripción: {art['description']}\n"
            existing_info += f"   Fuente: {art['source']}\n\n"
        
        # TODO: Crear prompt para LLM
        prompt = f"""
            Eres un experto en análisis de contenido de noticias. Compara el siguiente artículo NUEVO con los artículos EXISTENTES en la base de datos.

            ARTÍCULO NUEVO:
            Título: {new_title}
            Descripción: {new_description}

            ARTÍCULOS EXISTENTES EN BBDD (Top 3 más similares):
            {existing_info}

            INSTRUCCIONES:
            - TODO: ¿Qué debe analizar el LLM?
            - TODO: ¿Cómo debe evaluar similitud?
            - TODO: ¿Cuándo decidir SKIP vs INSERT?

            Responde SOLO con JSON válido:
            {{
                "decision": "insert" o "skip",
                "reasoning": "Explicación detallada de por qué tomaste esta decisión",
                "confidence": número entre 0.0 y 1.0
            }}
        """
        
        try:
            # TODO: Usar el LLM configurado
            response = Settings.llm.complete(prompt)
            result_text = response.text.strip()
            
            # TODO: Extraer JSON de la respuesta
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0].strip()
            elif "```" in result_text:
                result_text = result_text.split("```")[1].strip()
            
            try:
                llm_result = json.loads(result_text)
            except json.JSONDecodeError:
                # TODO: Fallback si JSON falla
                decision = "insert"
                reasoning = "Error parseando respuesta LLM, insertando por defecto"
                confidence = 0.5
                
                if "skip" in result_text.lower():
                    decision = "skip"
                    reasoning = "LLM sugirió skip pero hubo error en JSON"
                
                llm_result = {
                    "decision": decision,
                    "reasoning": reasoning,
                    "confidence": confidence
                }
            
            # TODO: Convertir decisión del LLM a enum
            decision = DeduplicationDecision.INSERT if llm_result["decision"].lower() == "insert" else DeduplicationDecision.SKIP
            
            return ArticleComparisonResult(
                decision=decision,
                reasoning=llm_result["reasoning"],
                confidence=float(llm_result["confidence"]),
                article_title=new_title
            )
            
        except Exception as e:
            # TODO: Fallback en caso de error
            return ArticleComparisonResult(
                decision=DeduplicationDecision.___________,  # TODO: ¿Decisión por defecto?
                reasoning=f"___________: {str(e)}",  # TODO: ¿Mensaje de error?
                confidence=___,  # TODO: ¿Confianza por defecto?
                article_title=new_title
            )
    
    def process_articles(self, new_articles: List[Dict], query: str) -> DeduplicationResult:
        """
        TODO: Procesamiento individual de artículos con flujo:
        1. Llamar API para obtener artículos
        2. Por cada artículo, buscar por título en BBDD
        3. LLM compara descripción nueva vs top 3 BBDD
        4. Decisión tipada individual por artículo
        """
        print("🤖 Evaluando deduplicación artículo por artículo con LLM...")
        
        if not new_articles:
            return DeduplicationResult(
                total_articles=0,
                articles_inserted=0,
                articles_skipped=0,
                individual_results=[]
            )
        
        individual_results = []
        articles_inserted = 0
        articles_skipped = 0
        
        for i, article in enumerate(new_articles, 1):
            article_title = article.get('title', f'Artículo {i}')
            print(f"📄 Procesando artículo {i}/{len(new_articles)}: {article_title[:50]}...")
            
            # TODO: PASO 1 - Buscar artículos similares por título en BBDD
            similar_articles = self.news_cache.search_similar_articles_by_title(article_title)
            print(f"   🔍 Encontrados {len(similar_articles)} artículos similares en BBDD")
            
            # TODO: PASO 2 - LLM compara descripción nueva vs top 3 BBDD
            comparison_result = self.compare_article_with_existing(article, similar_articles)
            individual_results.append(comparison_result)
            
            # TODO: PASO 3 - Acción basada en decisión LLM
            if comparison_result.decision == DeduplicationDecision.INSERT:
                # TODO: Insertar el artículo
                self.news_cache.insert_article(article, query)
                articles_inserted += 1
                print(f"   ✅ INSERTADO: {comparison_result.reasoning}")
                
                # TODO: Guardar pensamiento
                self.thought_manager.save_thought(
                    agent_name="NewsAgent",
                    query=f"{query} - {article_title[:30]}",
                    reasoning=comparison_result.reasoning,
                    decision="INSERT_ARTICLE",
                    confidence=comparison_result.confidence
                )
            else:
                articles_skipped += 1
                print(f"   🚫 OMITIDO: {comparison_result.reasoning}")
                
                # TODO: Guardar pensamiento
                self.thought_manager.save_thought(
                    agent_name="NewsAgent",
                    query=f"{query} - {article_title[:30]}",
                    reasoning=comparison_result.reasoning,
                    decision="SKIP_ARTICLE",
                    confidence=comparison_result.confidence
                )
        
        print(f"📊 Resumen: {articles_inserted} insertados, {articles_skipped} omitidos")
        
        return DeduplicationResult(
            total_articles=len(new_articles),
            articles_inserted=articles_inserted,
            articles_skipped=articles_skipped,
            individual_results=individual_results
        )

# =============================================================================
# CLASE 5: SERVICIO DE NOTICIAS (1.5 puntos)
# =============================================================================

class NewsService:
    """TODO: Servicio de noticias con deduplicación LLM"""
    
    def __init__(self, news_cache: NewsCache, deduplication_service: DeduplicationService):
        self.news_cache = news_cache
        self.deduplication_service = deduplication_service
    
    def search_news_with_deduplication(self, query: str, language: str = "es", page_size: int = 5) -> str:
        """TODO: Buscar noticias en internet y aplicar deduplicación LLM"""
        print(f"🔧 NewsService → search_news_with_deduplication(query={query}, language={language})")
        print(f"🌐 Buscando noticias en internet: {query}")
        
        try:
            # TODO: Siempre buscar en internet
            news_result = fetch_news(query=query, language=language, page_size=page_size)
            
            # TODO: Preparar datos para deduplicación
            articles_data = []
            for article in news_result.articles:
                articles_data.append({
                    'title': article.title,
                    'description': article.description,
                    'url': article.url,
                    'source': article.source,
                    'publishedAt': article.published_at
                })
            
            # TODO: Aplicar deduplicación LLM
            dedup_result = self.deduplication_service.process_articles(articles_data, query)
            
            # TODO: Formatear respuesta
            if dedup_result.articles_inserted > 0:
                cached_info = self.news_cache.search(query)
                response = f"📰 Noticias encontradas y añadidas al cache:\n\n"
                response += cached_info
            else:
                response = f"🔍 Información ya disponible en cache (artículos omitidos por duplicación):\n\n"
                cached_info = self.news_cache.search(query)
                response += cached_info + "\n\n"
                response += f"🤖 **Evaluación**: {dedup_result.articles_skipped} artículos omitidos por duplicación"
            
            print(f"✅ NewsService ← search_news_with_deduplication: {response[:100]}...")
            return response
            
        except Exception as e:
            error_msg = f"❌ Error buscando noticias: {str(e)}"
            print(f"✅ NewsService ← search_news_with_deduplication: {error_msg}")
            return error_msg

# =============================================================================
# CLASE 6: SERVICIO METEOROLÓGICO (1.5 puntos)
# =============================================================================

class WeatherService:
    """TODO: Servicio meteorológico con alertas de riesgo"""
    
    @staticmethod
    def calculate_fire_risk_index(temperature: float, humidity: float, wind: float = 0) -> str:
        """TODO: Calcular índice de riesgo de incendio"""
        print(f"🔧 WeatherService → calculate_fire_risk_index(temp={temperature}, humidity={humidity}, wind={wind})")
        risk_score = 0
        risk_factors = []
        
        # TODO: Evaluar temperatura
        if temperature > ___:  # TODO: ¿Temperatura crítica?
            risk_score += 3
            risk_factors.append("temperatura alta")
        elif temperature > ___:  # TODO: ¿Temperatura moderada?
            risk_score += 2
            risk_factors.append("temperatura moderada")
        elif temperature > ___:  # TODO: ¿Temperatura baja?
            risk_score += 1
        
        # TODO: Evaluar humedad
        if humidity < ___:  # TODO: ¿Humedad crítica?
            risk_score += 3
            risk_factors.append("humedad muy baja")
        elif humidity < ___:  # TODO: ¿Humedad moderada?
            risk_score += 2
            risk_factors.append("humedad baja")
        elif humidity < ___:  # TODO: ¿Humedad alta?
            risk_score += 1
        
        # TODO: Evaluar viento
        if wind > ___:  # TODO: ¿Viento crítico?
            risk_score += 2
            risk_factors.append("viento fuerte")
        elif wind > ___:  # TODO: ¿Viento moderado?
            risk_score += 1
            risk_factors.append("viento moderado")
        
        # TODO: Determinar nivel de riesgo
        if risk_score >= 6:
            risk_level = "MUY ALTO"
        elif risk_score >= 4:
            risk_level = "ALTO"
        elif risk_score >= 2:
            risk_level = "MODERADO"
        else:
            risk_level = "BAJO"
        
        result = f"🔥 Índice de riesgo de incendio: {risk_level} (puntuación: {risk_score}/8)\n"
        result += f"Factores: {', '.join(risk_factors) if risk_factors else 'Condiciones normales'}"
        
        print(f"✅ WeatherService ← calculate_fire_risk_index: {result[:50]}...")
        return result
    
    @staticmethod
    def get_weather_with_alerts(location: str) -> str:
        """TODO: Obtener clima con alertas de riesgo"""
        print(f"🔧 WeatherService → get_weather_with_alerts(location={location})")
        try:
            # TODO: Obtener datos del clima
            weather_data = fetch_current_weather(location)
            
            # TODO: Formatear respuesta básica
            response = f"En {location} hoy el clima es {weather_data.condition} "
            response += f"con una temperatura de {weather_data.temperature}°C "
            response += f"(sensación térmica de {weather_data.feels_like}°C). "
            response += f"La humedad es del {weather_data.humidity}% y hay un viento de {weather_data.wind_speed} m/s.\n\n"
            
            # TODO: Calcular y agregar índice de riesgo de incendio
            fire_risk = WeatherService.calculate_fire_risk_index(
                weather_data.temperature,
                weather_data.humidity,
                weather_data.wind_speed * 3.6  # Convertir m/s a km/h
            )
            response += fire_risk
            
            print(f"✅ WeatherService ← get_weather_with_alerts: {response[:100]}...")
            return response
            
        except Exception as e:
            error_msg = f"❌ Error obteniendo clima: {str(e)}"
            print(f"✅ WeatherService ← get_weather_with_alerts: {error_msg}")
            return error_msg

# =============================================================================
# CLASE 7: SISTEMA PRINCIPAL MULTI-AGENTE (1.5 puntos)
# =============================================================================

class MultiAgentSystem:
    """TODO: Sistema multi-agente coordinado"""
    
    def __init__(self):
        # TODO: Inicializar componentes
        self.thought_manager = AgentThoughtManager()
        self.news_cache = NewsCache()
        self.deduplication_service = DeduplicationService(self.news_cache, self.thought_manager)
        self.news_service = NewsService(self.news_cache, self.deduplication_service)
        self.weather_service = WeatherService()
        
        # Variables para agentes y workflow
        self.agents = {}
        self.workflow = None
    
    def initialize(self):
        """TODO: Inicializar el sistema completo"""
        print("🚀 Sistema Multi-Agente con APIs de Noticias y Clima")
        print("=" * 60)
        
        # TODO: 1. Configurar entorno
        print("\n1️⃣ Configurando entorno...")
        if not SystemConfig.verify_environment():
            return False
        
        SystemConfig.setup_llm_environment()
        
        # TODO: 2. Configurar cache
        print("\n2️⃣ Configurando cache de noticias...")
        self.news_cache.initialize()
        
        # TODO: 3. Crear agentes
        print("\n3️⃣ Creando agentes especializados...")
        self._create_agents()
        print("✅ Agentes creados")
        
        # TODO: 4. Configurar coordinación
        print("\n4️⃣ Configurando coordinación...")
        self._create_workflow()
        print("✅ Sistema de coordinación configurado")
        
        return True
    
    def _create_agents(self):
        """TODO: Crear agentes especializados"""
        # TODO: Herramientas de noticias
        news_search_tool = FunctionTool.from_defaults(
            fn=self.news_service.search_news_with_deduplication,
            name="___________",  # TODO: ¿Nombre de la herramienta?
            description="___________"  # TODO: ¿Descripción?
        )
        
        thoughts_tool = FunctionTool.from_defaults(
            fn=self.thought_manager.get_recent_thoughts,
            name="___________",  # TODO: ¿Nombre de la herramienta?
            description="___________"  # TODO: ¿Descripción?
        )
        
        # TODO: Herramientas meteorológicas
        weather_tool = FunctionTool.from_defaults(
            fn=self.weather_service.get_weather_with_alerts,
            name="___________",  # TODO: ¿Nombre de la herramienta?
            description="___________"  # TODO: ¿Descripción?
        )
        
        fire_risk_tool = FunctionTool.from_defaults(
            fn=self.weather_service.calculate_fire_risk_index,
            name="___________",  # TODO: ¿Nombre de la herramienta?
            description="___________"  # TODO: ¿Descripción?
        )
        
        # TODO: NewsAgent
        news_agent = FunctionAgent(
            name="___________",  # TODO: ¿Nombre del agente?
            description="___________",  # TODO: ¿Descripción?
            tools=[news_search_tool, thoughts_tool],
            system_prompt="___________"  # TODO: ¿System prompt?
        )
        
        # TODO: WeatherAgent
        weather_agent = FunctionAgent(
            name="___________",  # TODO: ¿Nombre del agente?
            description="___________",  # TODO: ¿Descripción?
            tools=[weather_tool, fire_risk_tool],
            system_prompt="___________"  # TODO: ¿System prompt?
        )
        
        # TODO: RouterAgent
        router_agent = FunctionAgent(
            name="___________",  # TODO: ¿Nombre del agente?
            description="___________",  # TODO: ¿Descripción?
            tools=[],
            can_handoff_to=["___________", "___________"],  # TODO: ¿A qué agentes puede derivar?
            system_prompt="""___________"""  # TODO: ¿System prompt completo?
        )
        
        self.agents = {
            "NewsAgent": news_agent,
            "WeatherAgent": weather_agent,
            "RouterAgent": router_agent
        }
    
    def _create_workflow(self):
        """TODO: Crear workflow de coordinación"""
        self.workflow = AgentWorkflow(
            agents=[
                self.agents["RouterAgent"],
                self.agents["NewsAgent"], 
                self.agents["WeatherAgent"]
            ],
            root_agent="___________",  # TODO: ¿Agente raíz?
            initial_state={}
        )
    
    async def run_tests(self):
        """TODO: Ejecutar casos de prueba del sistema"""
        print("\n5️⃣ Ejecutando casos de prueba...")
        
        test_cases = [
            "¿Cuáles son las últimas noticias sobre inteligencia artificial?",
            "Busca información sobre inteligencia artificial en las noticias",
            "¿Qué tiempo hace en Madrid hoy?",
            "¿Cuál es el índice de confort térmico en Barcelona con 25°C y 60% humedad?",
            "¿Hay noticias sobre tormentas en España?"
        ]
        
        for i, query in enumerate(test_cases, 1):
            print(f"\n🧪 Test {i}/{len(test_cases)}: {query}")
            print("-" * 60)
            
            try:
                # TODO: Ejecutar workflow
                handler = self.workflow.run(user_msg=query)
                async for event in handler.stream_events():
                    if hasattr(event, 'delta') and event.delta:
                        print(event.delta, end="", flush=True)
                
                print(f"\n✅ Test {i} completado")
                
            except Exception as e:
                print(f"❌ Error en test {i}: {str(e)}")
        
        print(f"\n🎉 ¡Todos los casos de prueba ejecutados!")
        print(f"\n📋 Resumen del sistema:")
        print(f"   ✅ Cache de noticias con ChromaDB")
        print(f"   ✅ NewsAgent con deduplicación LLM individual")
        print(f"   ✅ WeatherAgent con alertas de riesgo")
        print(f"   ✅ Router inteligente")
        print(f"   ✅ Sistema refactorizado con clases")

# =============================================================================
# FUNCIÓN PRINCIPAL
# =============================================================================

async def main():
    """TODO: Función principal del sistema"""
    try:
        # TODO: Crear e inicializar sistema
        system = MultiAgentSystem()
        
        if not system.initialize():
            return
        
        # TODO: Ejecutar casos de prueba
        await system.run_tests()
        
    except Exception as e:
        print(f"❌ Error general: {e}")
        print("🔍 Revisa tu configuración y variables de entorno")

if __name__ == "__main__":
    # TODO: Verificación rápida del entorno
    print("🔍 Verificando entorno...")
    
    required_packages = ['llama_index', 'openai', 'requests', 'chromadb', 'pydantic']
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - Instala con: pip install {package}")
    
    print("\n🚀 Iniciando sistema multi-agente...")
    
    # TODO: Ejecutar función principal
    asyncio.run(main())

# =============================================================================
# PISTAS PARA COMPLETAR EL TEMPLATE
# =============================================================================
"""
🔧 CONFIGURACIÓN:
- Modelo LLM: "gpt-4o-mini"
- Temperatura: 0.0 recomendada
- Variable OpenAI: "OPENAI_API_KEY"
- Modelo embeddings: "text-embedding-3-small"
- Text splitter: chunk_size=512, chunk_overlap=50

📁 PATHS Y NOMBRES:
- DB path: "./chroma_db"
- Colección: "news_cache"
- Top-k resultados: 3

🔄 ENUMS:
- DeduplicationDecision: INSERT = "insert", SKIP = "skip"

🌡️ RIESGO DE INCENDIO:
- Temperatura crítica: 30°C, moderada: 25°C, baja: 20°C
- Humedad crítica: 30%, moderada: 50%, alta: 70%
- Viento crítico: 20 km/h, moderado: 10 km/h

🛠️ HERRAMIENTAS:
- search_news: "Buscar noticias en internet con deduplicación automática LLM"
- get_agent_thoughts: "Consultar pensamientos y decisiones guardadas del agente"
- get_weather: "Obtener información meteorológica con alertas de riesgo"
- calculate_fire_risk: "Calcular índice de riesgo de incendio"

🤖 AGENTES:
- NewsAgent: "Especialista en noticias"
- WeatherAgent: "Especialista en información meteorológica"
- RouterAgent: "Coordinador principal del sistema"
- Agente raíz: "RouterAgent"

💭 SYSTEM PROMPTS:
- NewsAgent: "Eres un especialista en noticias. Usa search_news para buscar información actualizada en internet. La herramienta automáticamente maneja la deduplicación con LLM. Usa get_agent_thoughts para auditoría cuando sea necesario."
- WeatherAgent: "Eres un especialista en meteorología. Usa get_weather para obtener información del clima actual con alertas automáticas de riesgo de incendio. Puedes usar calculate_fire_risk para cálculos específicos."
- RouterAgent: "Eres el coordinador principal del sistema multi-agente. Tu función es: 1. **Consultas sobre NOTICIAS**: Deriva al NewsAgent 2. **Consultas sobre CLIMA/TIEMPO**: Deriva al WeatherAgent 3. **Consultas GENERALES**: Responde directamente"

¡Completa los TODO para implementar el sistema! 🚀
"""