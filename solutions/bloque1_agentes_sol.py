#!/usr/bin/env python3
"""
Sistema Multi-Agente con APIs de Noticias y Clima - Versión Refactorizada con Clases
Implementa deduplicación individual por artículo con comparación LLM
AUTOR: Sistema de IA - Curso LlamaIndex + Pydantic
FECHA: 2025
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
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

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
    """Decisión de deduplicación"""
    INSERT = "insert"
    SKIP = "skip"

class ArticleComparisonResult(BaseModel):
    """Resultado de comparación de un artículo individual"""
    decision: DeduplicationDecision = Field(description="Decisión: insertar o saltar")
    reasoning: str = Field(description="Razonamiento del LLM para este artículo")
    confidence: float = Field(description="Confianza de la decisión (0-1)", ge=0, le=1)
    article_title: str = Field(description="Título del artículo evaluado")

class DeduplicationResult(BaseModel):
    """Resultado de la evaluación de deduplicación completa"""
    total_articles: int = Field(description="Total de artículos procesados")
    articles_inserted: int = Field(description="Número de artículos insertados")
    articles_skipped: int = Field(description="Número de artículos omitidos")
    individual_results: List[ArticleComparisonResult] = Field(description="Resultados individuales por artículo")

class AgentThought(BaseModel):
    """Pensamiento del agente"""
    agent_name: str = Field(description="Nombre del agente")
    timestamp: str = Field(description="Momento de la decisión")
    query: str = Field(description="Consulta procesada")
    reasoning: str = Field(description="Razonamiento del agente")
    decision: str = Field(description="Decisión tomada")
    confidence: float = Field(description="Confianza (0-1)")

# =============================================================================
# CONFIGURACIÓN DEL SISTEMA
# =============================================================================

class SystemConfig:
    """Configuración global del sistema"""
    
    @staticmethod
    def verify_environment() -> bool:
        """Verificar que todas las dependencias estén instaladas"""
        print("🔍 Verificando entorno...")
        
        required_packages = [
            "llama_index", "openai", "requests", "chromadb", "pydantic"
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
        """Configurar el entorno LlamaIndex"""
        # Configurar LLM
        Settings.llm = OpenAI(
            model="gpt-4o-mini",
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Configurar embeddings
        Settings.embed_model = OpenAIEmbedding(
            model="text-embedding-3-small",
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Configurar text splitter
        Settings.text_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
        
        print("✅ Usando OpenAI GPT-4o-mini")

# =============================================================================
# GESTIÓN DE PENSAMIENTOS DE AGENTES
# =============================================================================

class AgentThoughtManager:
    """Gestor de pensamientos de agentes"""
    
    def __init__(self):
        self.thoughts: List[AgentThought] = []
    
    def save_thought(self, agent_name: str, query: str, reasoning: str, decision: str, confidence: float):
        """Guardar pensamiento del agente"""
        thought = AgentThought(
            agent_name=agent_name,
            timestamp=datetime.now().isoformat(),
            query=query,
            reasoning=reasoning,
            decision=decision,
            confidence=confidence
        )
        self.thoughts.append(thought)
        print(f"💭 {agent_name}: {decision} (confianza: {confidence:.2f})")
    
    def get_recent_thoughts(self, limit: int = 5) -> str:
        """Obtener pensamientos recientes del agente"""
        print(f"🔧 AgentThoughtManager → get_recent_thoughts(limit={limit})")
        
        if not self.thoughts:
            return "📝 No hay pensamientos guardados aún."
        
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
# CACHE DE NOTICIAS
# =============================================================================

class NewsCache:
    """Cache de noticias con ChromaDB"""
    
    def __init__(self, db_path: str = "./chroma_db"):
        self.db_path = db_path
        self.chroma_client = None
        self.news_collection = None
        self.news_index = None
        
    def initialize(self):
        """Inicializar cache de noticias con ChromaDB"""
        # Configurar ChromaDB
        self.chroma_client = chromadb.PersistentClient(
            path=self.db_path,
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        
        # Obtener o crear colección
        self.news_collection = self.chroma_client.get_or_create_collection(
            name="news_cache",
            metadata={"description": "Cache de noticias"}
        )
        
        # Crear vector store
        vector_store = ChromaVectorStore(chroma_collection=self.news_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        # Crear índice
        self.news_index = VectorStoreIndex([], storage_context=storage_context)
        
        print("✅ Cache de noticias configurado")
    
    def search(self, query: str) -> str:
        """Buscar en cache de noticias"""
        try:
            query_engine = self.news_index.as_query_engine(similarity_top_k=3)
            response = query_engine.query(query)
            return response.response if response.response else ""
        except Exception as e:
            print(f"⚠️ Error buscando en cache: {e}")
            return ""
    
    def search_similar_articles_by_title(self, title: str, top_k: int = 3) -> List[Dict]:
        """Buscar artículos similares por título en la BBDD"""
        if self.news_index is None:
            return []
        
        try:
            # Buscar por título específico
            retriever = self.news_index.as_retriever(similarity_top_k=top_k)
            nodes = retriever.retrieve(title)
            
            similar_articles = []
            for node in nodes:
                # Extraer información del nodo
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
        """Insertar un artículo en el cache"""
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
        self.news_index.insert(doc)

# =============================================================================
# SERVICIO DE DEDUPLICACIÓN LLM
# =============================================================================

class DeduplicationService:
    """Servicio de deduplicación usando LLM"""
    
    def __init__(self, news_cache: NewsCache, thought_manager: AgentThoughtManager):
        self.news_cache = news_cache
        self.thought_manager = thought_manager
    
    def compare_article_with_existing(self, new_article: Dict, similar_articles: List[Dict]) -> ArticleComparisonResult:
        """Comparar un artículo nuevo con artículos similares existentes usando LLM"""
        new_title = new_article.get('title', '')
        new_description = new_article.get('description', '')
        
        if not similar_articles:
            # No hay artículos similares, insertar
            return ArticleComparisonResult(
                decision=DeduplicationDecision.INSERT,
                reasoning="No se encontraron artículos similares en la base de datos",
                confidence=0.95,
                article_title=new_title
            )
        
        # Preparar información de artículos existentes
        existing_info = ""
        for i, art in enumerate(similar_articles[:3], 1):  # Top 3
            existing_info += f"{i}. Título: {art['title']}\n"
            existing_info += f"   Descripción: {art['description']}\n"
            existing_info += f"   Fuente: {art['source']}\n\n"
        
        prompt = f"""
Eres un experto en análisis de contenido de noticias. Compara el siguiente artículo NUEVO con los artículos EXISTENTES en la base de datos.

ARTÍCULO NUEVO:
Título: {new_title}
Descripción: {new_description}

ARTÍCULOS EXISTENTES EN BBDD (Top 3 más similares):
{existing_info}

INSTRUCCIONES:
- Analiza si el artículo NUEVO es sustancialmente diferente de los EXISTENTES
- Considera que la misma noticia puede venir de diferentes fuentes con ligeras variaciones
- Si el contenido es esencialmente el mismo (misma noticia, mismo evento), decide SKIP
- Si es una noticia diferente o un enfoque único, decide INSERT

Responde SOLO con JSON válido:
{{
    "decision": "insert" o "skip",
    "reasoning": "Explicación detallada de por qué tomaste esta decisión",
    "confidence": número entre 0.0 y 1.0
}}
"""
        
        try:
            response = Settings.llm.complete(prompt)
            result_text = response.text.strip()
            
            # Extraer JSON de la respuesta
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0].strip()
            elif "```" in result_text:
                result_text = result_text.split("```")[1].strip()
            
            try:
                llm_result = json.loads(result_text)
            except json.JSONDecodeError:
                # Intentar extraer campos manualmente si JSON falla
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
            
            decision = DeduplicationDecision.INSERT if llm_result["decision"].lower() == "insert" else DeduplicationDecision.SKIP
            
            return ArticleComparisonResult(
                decision=decision,
                reasoning=llm_result["reasoning"],
                confidence=float(llm_result["confidence"]),
                article_title=new_title
            )
            
        except Exception as e:
            # Fallback: insertar por defecto si hay error
            return ArticleComparisonResult(
                decision=DeduplicationDecision.INSERT,
                reasoning=f"Error en LLM, insertando por defecto: {str(e)}",
                confidence=0.5,
                article_title=new_title
            )
    
    def process_articles(self, new_articles: List[Dict], query: str) -> DeduplicationResult:
        """
        Procesamiento individual de artículos con flujo:
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
            
            # PASO 1: Buscar artículos similares por título en BBDD
            similar_articles = self.news_cache.search_similar_articles_by_title(article_title)
            print(f"   🔍 Encontrados {len(similar_articles)} artículos similares en BBDD")
            
            # PASO 2: LLM compara descripción nueva vs top 3 BBDD
            comparison_result = self.compare_article_with_existing(article, similar_articles)
            individual_results.append(comparison_result)
            
            # PASO 3: Acción basada en decisión LLM
            if comparison_result.decision == DeduplicationDecision.INSERT:
                # Insertar el artículo
                self.news_cache.insert_article(article, query)
                articles_inserted += 1
                print(f"   ✅ INSERTADO: {comparison_result.reasoning}")
                
                # Guardar pensamiento
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
                
                # Guardar pensamiento
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
# SERVICIO DE NOTICIAS
# =============================================================================

class NewsService:
    """Servicio de noticias con deduplicación LLM"""
    
    def __init__(self, news_cache: NewsCache, deduplication_service: DeduplicationService):
        self.news_cache = news_cache
        self.deduplication_service = deduplication_service
    
    def search_news_with_deduplication(self, query: str, language: str = "es", page_size: int = 5) -> str:
        """Buscar noticias en internet y aplicar deduplicación LLM"""
        print(f"🔧 NewsService → search_news_with_deduplication(query={query}, language={language})")
        print(f"🌐 Buscando noticias en internet: {query}")
        
        try:
            # Siempre buscar en internet
            news_result = fetch_news(query=query, language=language, page_size=page_size)
            
            # Preparar datos para deduplicación
            articles_data = []
            for article in news_result.articles:
                articles_data.append({
                    'title': article.title,
                    'description': article.description,
                    'url': article.url,
                    'source': article.source,
                    'publishedAt': article.published_at
                })
            
            # Aplicar deduplicación LLM
            dedup_result = self.deduplication_service.process_articles(articles_data, query)
            
            # Formatear respuesta
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
# SERVICIO METEOROLÓGICO
# =============================================================================

class WeatherService:
    """Servicio meteorológico con alertas de riesgo"""
    
    @staticmethod
    def calculate_fire_risk_index(temperature: float, humidity: float, wind: float = 0) -> str:
        """Calcular índice de riesgo de incendio"""
        print(f"🔧 WeatherService → calculate_fire_risk_index(temp={temperature}, humidity={humidity}, wind={wind})")
        risk_score = 0
        risk_factors = []
        
        # Evaluar temperatura
        if temperature > 30:
            risk_score += 3
            risk_factors.append("temperatura alta")
        elif temperature > 25:
            risk_score += 2
            risk_factors.append("temperatura moderada")
        elif temperature > 20:
            risk_score += 1
        
        # Evaluar humedad
        if humidity < 30:
            risk_score += 3
            risk_factors.append("humedad muy baja")
        elif humidity < 50:
            risk_score += 2
            risk_factors.append("humedad baja")
        elif humidity < 70:
            risk_score += 1
        
        # Evaluar viento
        if wind > 20:
            risk_score += 2
            risk_factors.append("viento fuerte")
        elif wind > 10:
            risk_score += 1
            risk_factors.append("viento moderado")
        
        # Determinar nivel de riesgo
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
        """Obtener clima con alertas de riesgo"""
        print(f"🔧 WeatherService → get_weather_with_alerts(location={location})")
        try:
            # Obtener datos del clima
            weather_data = fetch_current_weather(location)
            
            # Formatear respuesta básica
            response = f"En {location} hoy el clima es {weather_data.condition} "
            response += f"con una temperatura de {weather_data.temperature}°C "
            response += f"(sensación térmica de {weather_data.feels_like}°C). "
            response += f"La humedad es del {weather_data.humidity}% y hay un viento de {weather_data.wind_speed} m/s.\n\n"
            
            # Calcular y agregar índice de riesgo de incendio
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
# SISTEMA DE AGENTES
# =============================================================================

class MultiAgentSystem:
    """Sistema multi-agente coordinado"""
    
    def __init__(self):
        # Inicializar componentes
        self.thought_manager = AgentThoughtManager()
        self.news_cache = NewsCache()
        self.deduplication_service = DeduplicationService(self.news_cache, self.thought_manager)
        self.news_service = NewsService(self.news_cache, self.deduplication_service)
        self.weather_service = WeatherService()
        
        # Variables para agentes y workflow
        self.agents = {}
        self.workflow = None
    
    def initialize(self):
        """Inicializar el sistema completo"""
        print("🚀 Sistema Multi-Agente con APIs de Noticias y Clima")
        print("=" * 60)
        
        # 1. Configurar entorno
        print("\n1️⃣ Configurando entorno...")
        if not SystemConfig.verify_environment():
            return False
        
        SystemConfig.setup_llm_environment()
        
        # 2. Configurar cache
        print("\n2️⃣ Configurando cache de noticias...")
        self.news_cache.initialize()
        
        # 3. Crear agentes
        print("\n3️⃣ Creando agentes especializados...")
        self._create_agents()
        print("✅ Agentes creados")
        
        # 4. Configurar coordinación
        print("\n4️⃣ Configurando coordinación...")
        self._create_workflow()
        print("✅ Sistema de coordinación configurado")
        
        return True
    
    def _create_agents(self):
        """Crear agentes especializados"""
        # Herramientas de noticias
        news_search_tool = FunctionTool.from_defaults(
            fn=self.news_service.search_news_with_deduplication,
            name="search_news",
            description="Buscar noticias en internet con deduplicación automática LLM"
        )
        
        thoughts_tool = FunctionTool.from_defaults(
            fn=self.thought_manager.get_recent_thoughts,
            name="get_agent_thoughts",
            description="Consultar pensamientos y decisiones guardadas del agente"
        )
        
        # Herramientas meteorológicas
        weather_tool = FunctionTool.from_defaults(
            fn=self.weather_service.get_weather_with_alerts,
            name="get_weather",
            description="Obtener información meteorológica con alertas de riesgo"
        )
        
        fire_risk_tool = FunctionTool.from_defaults(
            fn=self.weather_service.calculate_fire_risk_index,
            name="calculate_fire_risk",
            description="Calcular índice de riesgo de incendio"
        )
        
        # NewsAgent
        news_agent = FunctionAgent(
            name="NewsAgent",
            description="Especialista en noticias",
            tools=[news_search_tool, thoughts_tool],
            system_prompt="Eres un especialista en noticias. Usa search_news para buscar información actualizada en internet. La herramienta automáticamente maneja la deduplicación con LLM. Usa get_agent_thoughts para auditoría cuando sea necesario."
        )
        
        # WeatherAgent
        weather_agent = FunctionAgent(
            name="WeatherAgent",
            description="Especialista en información meteorológica",
            tools=[weather_tool, fire_risk_tool],
            system_prompt="Eres un especialista en meteorología. Usa get_weather para obtener información del clima actual con alertas automáticas de riesgo de incendio. Puedes usar calculate_fire_risk para cálculos específicos."
        )
        
        # RouterAgent
        router_agent = FunctionAgent(
            name="RouterAgent",
            description="Coordinador principal del sistema",
            tools=[],
            can_handoff_to=["NewsAgent", "WeatherAgent"],
            system_prompt="""Eres el coordinador principal del sistema multi-agente. Tu función es:

1. **Consultas sobre NOTICIAS**: Deriva al NewsAgent
2. **Consultas sobre CLIMA/TIEMPO**: Deriva al WeatherAgent  
3. **Consultas GENERALES**: Responde directamente

Ejemplos:
- "noticias sobre IA" → NewsAgent
- "tiempo en Madrid" → WeatherAgent
- "¿cómo funciona el sistema?" → Respuesta directa

Sé claro y conciso en tus derivaciones."""
        )
        
        self.agents = {
            "NewsAgent": news_agent,
            "WeatherAgent": weather_agent,
            "RouterAgent": router_agent
        }
    
    def _create_workflow(self):
        """Crear workflow de coordinación"""
        self.workflow = AgentWorkflow(
            agents=[
                self.agents["RouterAgent"],
                self.agents["NewsAgent"], 
                self.agents["WeatherAgent"]
            ],
            root_agent="RouterAgent",
            initial_state={}
        )
    
    async def run_tests(self):
        """Ejecutar casos de prueba del sistema"""
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
    """Función principal del sistema"""
    try:
        # Crear e inicializar sistema
        system = MultiAgentSystem()
        
        if not system.initialize():
            return
        
        # Ejecutar casos de prueba
        await system.run_tests()
        
    except Exception as e:
        print(f"❌ Error general: {e}")
        print("🔍 Revisa tu configuración y variables de entorno")

if __name__ == "__main__":
    asyncio.run(main())
