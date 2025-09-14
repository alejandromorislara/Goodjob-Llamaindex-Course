"""
🎵 Solución: Parseo Avanzado con Análisis de Letras y Base de Datos Vectorial

Esta es la implementación funcional completa del ejercicio de parseo musical
con extracción de letras usando LLM tipado y persistencia en ChromaDB.
Nota : Usar un modelo de IA mas potente que gpt-4o-mini para la extracción de letras. Para la extracción de texto seria recomendable
usar técnicas de web scraping como se comentó en el notebook y durante el curso.

Autor: Sistema de IA - Curso LlamaIndex + Pydantic
Fecha: 2025
"""

import os
import sys
import json
import re
from datetime import datetime
from typing import List, Dict, Optional, Union, Any
from pathlib import Path
import requests
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Pydantic imports
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic import ValidationError
from dotenv import load_dotenv
from enum import Enum

# Agregar src al path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Cargar variables de entorno
load_dotenv()

# LlamaIndex imports
from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.program import LLMTextCompletionProgram
from llama_index.readers.web import SimpleWebPageReader

# =============================================================================
# ENUMS Y TIPOS BASE
# =============================================================================

class SentimentType(str, Enum):
    """Tipos de sentimiento para análisis de letras"""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    MELANCHOLIC = "melancholic"
    ENERGETIC = "energetic"
    ROMANTIC = "romantic"
    ANGRY = "angry"

class LanguageType(str, Enum):
    """Idiomas detectados en las letras"""
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    ITALIAN = "it"
    UNKNOWN = "unknown"

# =============================================================================
# MODELOS PYDANTIC IMPLEMENTADOS
# =============================================================================

class WebLyricsData(BaseModel):
    """Modelo para datos extraídos de la web con LLM"""
    
    lyrics: str = Field(..., description="Letras completas(lyrics) extraídas de la página web")
    views: int = Field(default=0, ge=0, description="Número de visualizaciones")
    composers: List[str] = Field(default_factory=list, description="Compositores de la canción")
    additional_info: Optional[str] = Field(None, description="Información adicional relevante")
    source_url: str = Field(..., description="URL de origen de los datos")

class LyricsAnalysis(BaseModel):
    """Modelo para análisis de letras con LLM implementado"""
    
    lyrics: str = Field(..., description="Letras completas de la canción")
    sentiment: SentimentType = Field(..., description="Sentimiento principal de la canción")
    language: LanguageType = Field(..., description="Idioma detectado de las letras")
    themes: List[str] = Field(default_factory=list, description="Temas principales (máximo 5)")
    emotional_intensity: float = Field(..., ge=0.0, le=1.0, description="Intensidad emocional (0-1)")
    word_count: int = Field(default=0, ge=0, description="Número de palabras en las letras")
    views: int = Field(default=0, ge=0, description="Número de visualizaciones de la página web")
    composers: List[str] = Field(default_factory=list, description="Compositores de la canción")
    web_source: Optional[str] = Field(None, description="URL de origen de los datos")
    
    @field_validator('themes')
    @classmethod
    def validate_themes(cls, v):
        """Validar que no haya más de 5 temas"""
        if len(v) > 5:
            v = v[:5]  # Truncar a 5 temas
        return [theme.strip().lower() for theme in v if theme.strip()]
    
    @field_validator('lyrics')
    @classmethod
    def validate_lyrics(cls, v):
        """Validar que las letras no estén vacías"""
        if not v or not v.strip():
            raise ValueError("Las letras no pueden estar vacías")
        return v.strip()
    
    @model_validator(mode='after')
    def calculate_word_count(self):
        """Calcular automáticamente el número de palabras"""
        if self.lyrics:
            self.word_count = len(self.lyrics.split())
        return self

class Artist(BaseModel):
    """Modelo Pydantic para artista implementado"""
    
    name: str = Field(..., description="Nombre del artista")
    genre: str = Field(..., description="Género musical principal")
    description: Optional[str] = Field(None, description="Descripción del artista")
    image_url: Optional[str] = Field(None, description="URL de la imagen del artista")
    url: Optional[str] = Field(None, description="URL oficial del artista")
    
    @field_validator('genre')
    @classmethod
    def validate_genre(cls, v):
        """Validar que el género esté en la lista permitida"""
        allowed_genres = ["Rock Alternativo", "Rock", "Electronic", "Experimental"]
        if v not in allowed_genres:
            print(f"⚠️ Género '{v}' no está en la lista permitida, pero se acepta")
        return v
    
    @field_validator('description')
    @classmethod
    def sanitize_description(cls, v):
        """Sanitizar descripción eliminando caracteres especiales"""
        if v:
            return re.sub(r'[^\w\s\-.,]', '', v)
        return v
    
    @field_validator('name')
    @classmethod
    def normalize_name(cls, v):
        """Normalizar nombre a Title Case"""
        return v.title() if v else v

class Album(BaseModel):
    """Modelo Pydantic para álbumes implementado"""
    
    id: str = Field(..., description="ID único del álbum")
    name: str = Field(..., description="Nombre del álbum")
    date_published: int = Field(..., description="Año de publicación")
    description: Optional[str] = Field(None, description="Descripción del álbum")
    genre: str = Field(..., description="Género musical")
    image_url: Optional[str] = Field(None, description="URL de la imagen del álbum")
    num_tracks: int = Field(..., description="Número de tracks")
    publisher: str = Field(..., description="Editorial/Sello discográfico")
    url: str = Field(..., description="URL del álbum")
    
    @field_validator('date_published')
    @classmethod
    def validate_publication_date(cls, v):
        if not (1900 <= v <= 2030):
            raise ValueError("Fecha de publicación debe estar entre 1900-2030")
        return v
    
    @field_validator('num_tracks')
    @classmethod
    def validate_num_tracks(cls, v):
        if not (1 <= v <= 50):
            raise ValueError("Número de tracks debe estar entre 1-50")
        return v
    
    @field_validator('name')
    @classmethod
    def normalize_name(cls, v):
        return v.title() if v else v
    
    @field_validator('publisher')
    @classmethod
    def validate_publisher(cls, v):
        if not v or not v.strip():
            raise ValueError("Editorial no puede estar vacía")
        return v

class Song(BaseModel):
    """Modelo Pydantic para canciones con análisis de letras implementado"""
    
    id: str = Field(..., description="ID único de la canción")
    name: str = Field(..., description="Nombre de la canción")
    description: Optional[str] = Field(None, description="Descripción de la canción")
    duration_iso: str = Field(..., description="Duración en formato ISO 8601")
    duration_formatted: Optional[str] = Field(None, description="Duración en formato MM:SS")
    genre: str = Field(..., description="Género musical")
    image_url: Optional[str] = Field(None, description="URL de la imagen")
    url: str = Field(..., description="URL de la canción")
    album_name: str = Field(..., description="Nombre del álbum")
    album_id: str = Field(..., description="ID del álbum")
    artist_name: str = Field(..., description="Nombre del artista")
    artist_id: str = Field(..., description="ID del artista")
    lyrics_analysis: Optional[LyricsAnalysis] = Field(None, description="Análisis de letras con LLM")
    
    @field_validator('duration_iso')
    @classmethod
    def validate_duration_iso(cls, v):
        pattern = r'^PT\d+M\d+S$'
        if not re.match(pattern, v):
            raise ValueError("Duración debe estar en formato ISO 8601 (PT3M55S)")
        return v
    
    @model_validator(mode='after')
    def format_duration(self):
        if self.duration_iso and not self.duration_formatted:
            duration_iso = self.duration_iso
            if duration_iso.startswith('PT') and 'M' in duration_iso and 'S' in duration_iso:
                minutes = int(duration_iso.split('M')[0][2:])
                seconds = int(duration_iso.split('M')[1].replace('S', ''))
                self.duration_formatted = f"{minutes}:{seconds:02d}"
        return self
    
    @field_validator('name')
    @classmethod
    def sanitize_name(cls, v):
        return re.sub(r'[^\w\s\-.,()]', '', v) if v else v

# =============================================================================
# CLASE PARSER IMPLEMENTADA
# =============================================================================

class RadioheadParser:
    """Parser principal del documento JSON implementado"""
    
    def __init__(self, file_path: str = "data/radiohead.json"):
        self.file_path = file_path
        self.raw_data: Optional[Dict] = None
        self.artist: Optional[Artist] = None
        self.albums: List[Album] = []
        self.songs: List[Song] = []
    
    def load_json(self) -> Dict:
        print(f"📁 Cargando {self.file_path}...")
        
        try:
            with open(self.file_path, 'r', encoding='utf-8') as file:
                self.raw_data = json.load(file)
            
            required_keys = ['@context', '@type', 'name', 'track']
            for key in required_keys:
                if key not in self.raw_data:
                    raise ValueError(f"Clave requerida '{key}' no encontrada en JSON")
            
            print("✅ JSON cargado correctamente")
            return self.raw_data
            
        except FileNotFoundError:
            print(f"❌ Archivo {self.file_path} no encontrado")
            raise
        except json.JSONDecodeError as e:
            print(f"❌ Error al parsear JSON: {e}")
            raise
        except Exception as e:
            print(f"❌ Error inesperado: {e}")
            raise
    
    def extract_artist_info(self) -> Artist:
        if not self.raw_data:
            raise ValueError("Debe cargar el JSON primero")
        
        print("🎤 Extrayendo información del artista...")
        
        artist_data = {
            'name': self.raw_data.get('name', ''),
            'genre': self.raw_data.get('genre', ''),
            'description': self.raw_data.get('description', ''),
            'image_url': self.raw_data.get('image', ''),
            'url': self.raw_data.get('url', '')
        }
        
        self.artist = Artist(**artist_data)
        
        print(f"✅ Artista extraído: {self.artist.name}")
        return self.artist
    
    def extract_albums(self) -> List[Album]:
        if not self.raw_data:
            raise ValueError("Debe cargar el JSON primero")
        
        print("💿 Extrayendo álbumes...")
        
        albums_dict = {}
        
        for track in self.raw_data.get('track', []):
            album_info = track.get('inAlbum', {})
            album_id = album_info.get('@id', '')
            
            if album_id and album_id not in albums_dict:
                album_data = {
                    'id': album_id,
                    'name': album_info.get('name', ''),
                    'date_published': album_info.get('datePublished', 0),
                    'description': album_info.get('description', ''),
                    'genre': album_info.get('genre', ''),
                    'image_url': album_info.get('image', ''),
                    'num_tracks': album_info.get('numtracks', 0),
                    'publisher': album_info.get('publisher', ''),
                    'url': album_info.get('url', '')
                }
                
                try:
                    albums_dict[album_id] = Album(**album_data)
                except ValidationError as e:
                    print(f"⚠️ Error validando álbum {album_data.get('name', 'N/A')}: {e}")
        
        self.albums = list(albums_dict.values())
        print(f"✅ {len(self.albums)} álbumes extraídos")
        return self.albums
    
    def extract_songs(self) -> List[Song]:
        if not self.raw_data:
            raise ValueError("Debe cargar el JSON primero")
        
        print("🎵 Extrayendo canciones...")
        
        songs = []
        
        for track in self.raw_data.get('track', []):
            song_data = {
                'id': track.get('@id', ''),
                'name': track.get('name', ''),
                'description': track.get('description', ''),
                'duration_iso': track.get('duration', ''),
                'genre': track.get('genre', ''),
                'image_url': track.get('image', ''),
                'url': track.get('url', ''),
                'album_name': track.get('inAlbum', {}).get('name', ''),
                'album_id': track.get('inAlbum', {}).get('@id', ''),
                'artist_name': track.get('byArtist', {}).get('name', ''),
                'artist_id': track.get('byArtist', {}).get('@id', '')
            }
            
            try:
                songs.append(Song(**song_data))
            except ValidationError as e:
                print(f"⚠️ Error validando canción {song_data.get('name', 'N/A')}: {e}")
        
        self.songs = songs
        print(f"✅ {len(self.songs)} canciones extraídas")
        return self.songs

# =============================================================================
# CLASE EXTRACTOR DE LETRAS WEB IMPLEMENTADA
# =============================================================================

class WebLyricsExtractor:
    """Extractor de letras desde páginas web usando SimpleWebPageReader y LLM tipado"""
    
    def __init__(self):
        # Configurar LlamaIndex con parámetros optimizados para JSON
        Settings.llm = OpenAI(
            model="gpt-4o-mini", 
            temperature=0.0,
            additional_kwargs={
                "response_format": {"type": "json_object"}
            }
        )
        Settings.embed_model = OpenAIEmbedding()
        self.llm = Settings.llm
        self.web_reader = SimpleWebPageReader()
    
    def extract_lyrics_from_url(self, url: str, song_name: str, artist_name: str) -> Optional[WebLyricsData]:
        """Extraer letras y datos desde una URL usando SimpleWebPageReader y LLM tipado"""
        print(f"🌐 Extrayendo datos de: {url}")
        
        try:
                        # Leer contenido de la página web
            documents = self.web_reader.load_data([url])
            
            if not documents:
                print(f"❌ No se pudo cargar contenido de {url}")
                return None
            
            # Obtener el contenido de la página
            page_content = documents[0].text
            
            # Crear prompt para extraer datos estructurados
            prompt_template = """
            Extrae información de esta página web de letras de canciones y devuélvela en el formato JSON especificado.
            
            Contenido de la página:
            {page_content}
            
            Canción: {song_name}
            Artista: {artist_name}
            URL: {url}
            
            Extrae la siguiente información y devuélvela como un objeto JSON válido:
            - lyrics: La primera frase de las letras de la canción
            - views: El número de visualizaciones (busca números como "290,298 visualizaciones" o similares)
            - composers: Lista de compositores (busca "Compuesta por:" o similar)
            - additional_info: Cualquier información adicional relevante
            - source_url: La URL proporcionada ({url})
            
            IMPORTANTE: Devuelve ÚNICAMENTE el objeto JSON, sin texto adicional, sin markdown, sin explicaciones.
            """
            
            # Usar LLM con output tipado
            program = LLMTextCompletionProgram.from_defaults(
                output_cls=WebLyricsData,
                prompt_template_str=prompt_template,
                llm=self.llm
            )
            
            result = program(
                page_content=page_content[:1000],  # Limit to the first 3000 characters to avoid excessive tokens
                song_name=song_name,
                artist_name=artist_name,
                url=url
            )
            
            print(f"✅ Datos extraídos de {url}")
            return result
            
        except Exception as e:
            print(f"❌ Error extrayendo datos de {url}: {e}")
            # Crear datos de fallback estructurados para mantener tipado
            try:
                fallback_data = WebLyricsData(
                    lyrics=f"[Letras no disponibles para {song_name} - {artist_name}]",
                    views=0,
                    composers=[],
                    additional_info=f"Error de extracción: {str(e)[:100]}",
                    source_url=url
                )
                print(f"⚠️ Usando datos de fallback para {song_name}.Page content {page_content[:1000]}")
                return fallback_data
            except Exception as fallback_error:
                print(f"❌ Error creando fallback: {fallback_error}")
                return None
    
    def analyze_lyrics_with_web_data(self, web_data: WebLyricsData, song_name: str, artist_name: str) -> Optional[LyricsAnalysis]:
        """Analizar letras extraídas de la web y crear análisis completo"""
        print(f"🎤 Analizando letras de: {song_name} - {artist_name}")
        
        try:
            # Crear prompt para análisis de sentimiento y temas
            prompt_template = """
            Analiza las letras de la canción "{song_name}" de {artist_name} y proporciona un análisis estructurado.
            
            Letras de la canción:
            {lyrics}
            
            Información adicional:
            - Visualizaciones: {views}
            - Compositores: {composers}
            
            Proporciona un análisis completo en formato JSON con los siguientes campos:
            - lyrics: Las letras completas (copia exacta del input)
            - sentiment: Uno de estos valores exactos: "positive", "negative", "neutral", "melancholic", "energetic", "romantic", "angry"
            - language: Código de idioma: "en", "es", "fr", "it", "unknown"
            - themes: Lista de máximo 5 temas principales (como strings)
            - emotional_intensity: Número decimal entre 0.0 y 1.0
            - word_count: Número entero de palabras en las letras
            - views: El número de visualizaciones (usar el valor proporcionado: {views})
            - composers: Lista de compositores (usar la lista proporcionada)
            - web_source: URL de origen (dejar como null por ahora)
            
            IMPORTANTE: Devuelve ÚNICAMENTE el objeto JSON válido, sin markdown, sin explicaciones adicionales.
            """
            
            # Usar LLM con output tipado
            program = LLMTextCompletionProgram.from_defaults(
                output_cls=LyricsAnalysis,
                prompt_template_str=prompt_template,
                llm=self.llm
            )
            
            result = program(
                song_name=song_name,
                artist_name=artist_name,
                lyrics=web_data.lyrics,
                views=web_data.views,
                composers=", ".join(web_data.composers) if web_data.composers else "N/A"
            )
            
            # Agregar datos web al análisis
            result.views = web_data.views
            result.composers = web_data.composers
            result.web_source = web_data.source_url
            
            print(f"✅ Análisis completado para {song_name}")
            return result
            
        except Exception as e:
            print(f"❌ Error analizando letras de {song_name}: {e}")
            # Crear análisis de fallback estructurado para mantener tipado
            try:
                fallback_analysis = LyricsAnalysis(
                    lyrics=web_data.lyrics,
                    sentiment=SentimentType.NEUTRAL,
                    language=LanguageType.ENGLISH,
                    themes=["rock", "alternative"],
                    emotional_intensity=0.5,
                    word_count=len(web_data.lyrics.split()) if web_data.lyrics else 0,
                    views=web_data.views,
                    composers=web_data.composers,
                    web_source=web_data.source_url
                )
                print(f"⚠️ Usando análisis de fallback para {song_name}")
                return fallback_analysis
            except Exception as fallback_error:
                print(f"❌ Error creando análisis de fallback: {fallback_error}")
                return None
    
    def process_song_with_web_url(self, song: Song, letras_url: str) -> Song:
        """Procesar una canción completa con su URL de letras.com"""
        print(f"🎵 Procesando {song.name} con URL: {letras_url}")
        
        # Extraer datos de la web
        web_data = self.extract_lyrics_from_url(letras_url, song.name, song.artist_name)
        
        if web_data:
            # Analizar letras
            lyrics_analysis = self.analyze_lyrics_with_web_data(web_data, song.name, song.artist_name)
            
            if lyrics_analysis:
                song.lyrics_analysis = lyrics_analysis
                print(f"✅ {song.name} procesada exitosamente")
            else:
                print(f"⚠️ No se pudo analizar las letras de {song.name}")
        else:
            print(f"⚠️ No se pudieron extraer datos web de {song.name}")
        
        return song

# =============================================================================
# CLASE BASE DE DATOS VECTORIAL IMPLEMENTADA
# =============================================================================

class VectorMusicDatabase:
    """Base de datos vectorial persistente para música implementada"""
    
    def __init__(self, persist_dir: str = "./chroma_music_db"):
        self.persist_dir = persist_dir
        self.collection_name = "radiohead_songs"
        self.client = None
        self.collection = None
        
        # Configurar LlamaIndex
        Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0.0)
        Settings.embed_model = OpenAIEmbedding()
    
    def initialize_database(self):
        """Inicializar base de datos vectorial persistente"""
        print("🗄️ Inicializando base de datos vectorial...")
        
        try:
            # Importar ChromaDB
            import chromadb
            from chromadb.config import Settings as ChromaSettings
            
            # Crear cliente persistente
            self.client = chromadb.PersistentClient(
                path=self.persist_dir,
                settings=ChromaSettings(anonymized_telemetry=False)
            )
            
            # Crear o obtener colección
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "Radiohead songs with web-extracted lyrics and sentiment analysis"}
            )
            
            print("✅ Base de datos vectorial inicializada")
            
        except Exception as e:
            print(f"❌ Error inicializando base de datos: {e}")
            raise
    
    def save_songs_to_vector_db(self, songs: List[Song]):
        """Guardar canciones con letras extraídas de la web en base de datos vectorial"""
        if not self.collection:
            raise ValueError("Debe inicializar la base de datos primero")
        
        print(f"💾 Guardando {len(songs)} canciones en base de datos vectorial...")
        
        documents = []
        metadatas = []
        ids = []
        
        for song in songs:
            if song.lyrics_analysis:
                # Crear documento con letras y metadata web
                doc_text = f"""
                Canción: {song.name}
                Artista: {song.artist_name}
                Álbum: {song.album_name}
                Género: {song.genre}
                Duración: {song.duration_formatted}
                Visualizaciones: {song.lyrics_analysis.views:,}
                Compositores: {', '.join(song.lyrics_analysis.composers)}
                
                Letras:
                {song.lyrics_analysis.lyrics}
                """
                
                # Crear metadata completa con datos web
                metadata = {
                    "song_id": song.id,
                    "song_name": song.name,
                    "artist_name": song.artist_name,
                    "album_name": song.album_name,
                    "genre": song.genre,
                    "duration": song.duration_formatted,
                    "sentiment": song.lyrics_analysis.sentiment.value,
                    "language": song.lyrics_analysis.language.value,
                    "emotional_intensity": song.lyrics_analysis.emotional_intensity,
                    "word_count": song.lyrics_analysis.word_count,
                    "themes": ",".join(song.lyrics_analysis.themes),
                    "views": song.lyrics_analysis.views,
                    "composers": ",".join(song.lyrics_analysis.composers),
                    "web_source": song.lyrics_analysis.web_source or "",
                    "url": song.url
                }
                
                # Agregar a listas
                documents.append(doc_text.strip())
                metadatas.append(metadata)
                ids.append(f"song_{hash(song.id)}")
        
        try:
            # Insertar en ChromaDB
            if documents:
                self.collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
            
            print(f"✅ {len(documents)} canciones guardadas en base de datos vectorial")
            
        except Exception as e:
            print(f"❌ Error guardando en base de datos vectorial: {e}")
            raise
    
    def search_songs_by_views(self, min_views: int = 100000, limit: int = 5) -> List[Dict]:
        """Buscar canciones por número mínimo de visualizaciones"""
        if not self.collection:
            raise ValueError("Debe inicializar la base de datos primero")
        
        print(f"🔍 Buscando canciones con más de {min_views:,} visualizaciones")
        
        try:
            # Obtener todas las canciones y filtrar por visualizaciones
            all_results = self.collection.get()
            
            if not all_results or 'metadatas' not in all_results:
                return {"ids": [[]], "metadatas": [[]], "documents": [[]]}
            
            # Filtrar por visualizaciones
            filtered_ids = []
            filtered_metadatas = []
            filtered_documents = []
            
            for i, metadata in enumerate(all_results['metadatas']):
                views = int(metadata.get('views', 0))
                if views >= min_views:
                    filtered_ids.append(all_results['ids'][i])
                    filtered_metadatas.append(metadata)
                    if 'documents' in all_results:
                        filtered_documents.append(all_results['documents'][i])
            
            # Limitar resultados
            filtered_ids = filtered_ids[:limit]
            filtered_metadatas = filtered_metadatas[:limit]
            filtered_documents = filtered_documents[:limit]
            
            return {
                "ids": [filtered_ids],
                "metadatas": [filtered_metadatas],
                "documents": [filtered_documents]
            }
            
        except Exception as e:
            print(f"❌ Error buscando por visualizaciones: {e}")
            return {"ids": [[]], "metadatas": [[]], "documents": [[]]}
    
    def get_statistics(self) -> Dict[str, Any]:
        """Obtener estadísticas completas incluyendo datos web"""
        if not self.collection:
            raise ValueError("Debe inicializar la base de datos primero")
        
        print("📊 Obteniendo estadísticas de base de datos vectorial...")
        
        try:
            count = self.collection.count()
            
            # Obtener todas las canciones para estadísticas
            all_songs = self.collection.get()
            
            # Estadísticas por sentimiento
            sentiment_stats = {}
            for sentiment in SentimentType:
                sentiment_count = 0
                if all_songs and 'metadatas' in all_songs:
                    for metadata in all_songs['metadatas']:
                        if metadata.get('sentiment') == sentiment.value:
                            sentiment_count += 1
                sentiment_stats[sentiment.value] = sentiment_count
            
            # Estadísticas adicionales con datos web
            all_songs = self.collection.get()
            albums = set()
            languages = set()
            total_intensity = 0
            total_views = 0
            composers_set = set()
            
            if all_songs and 'metadatas' in all_songs:
                for metadata in all_songs['metadatas']:
                    albums.add(metadata.get('album_name', 'Unknown'))
                    languages.add(metadata.get('language', 'unknown'))
                    total_intensity += float(metadata.get('emotional_intensity', 0))
                    total_views += int(metadata.get('views', 0))
                    
                    # Agregar compositores
                    composers_str = metadata.get('composers', '')
                    if composers_str:
                        composers_set.update([c.strip() for c in composers_str.split(',') if c.strip()])
            
            avg_intensity = total_intensity / count if count > 0 else 0
            avg_views = total_views / count if count > 0 else 0
            
            return {
                "total_songs": count,
                "total_albums": len(albums),
                "languages_detected": list(languages),
                "sentiment_distribution": sentiment_stats,
                "average_emotional_intensity": round(avg_intensity, 2),
                "total_views": total_views,
                "average_views": round(avg_views, 0),
                "unique_composers": len(composers_set),
                "collection_name": self.collection_name,
                "persist_dir": self.persist_dir
            }
            
        except Exception as e:
            print(f"❌ Error obteniendo estadísticas: {e}")
            return {}

# =============================================================================
# FUNCIÓN PRINCIPAL DE DEMOSTRACIÓN
# =============================================================================

def test_web_lyrics_extraction():
    """Función de prueba de extracción de letras desde web"""
    print("🧪 Iniciando pruebas de extracción web...")
    
    # Verificar API key
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️ OPENAI_API_KEY no configurada, saltando extracción web")
        return []
    
    try:
        # URLs de ejemplo de letras.com para Radiohead
        test_urls = {
            "Let Down": "https://www.letras.com/radiohead/79031/",
            "Creep": "https://www.letras.com/radiohead/78989/",
            "Karma Police": "https://www.letras.com/radiohead/79030/"
        }
        
        # Inicializar extractor web
        web_extractor = WebLyricsExtractor()
        
        # Parsear canciones básicas
        parser = RadioheadParser()
        parser.load_json()
        songs = parser.extract_songs()
        
        # Procesar algunas canciones con URLs web
        processed_songs = []
        
        for song in songs[:3]:  # Solo las primeras 3 para prueba
            if song.name in test_urls:
                url = test_urls[song.name]
                processed_song = web_extractor.process_song_with_web_url(song, url)
                processed_songs.append(processed_song)
            else:
                # Para canciones sin URL específica, usar una URL genérica
                generic_url = f"https://www.letras.com/radiohead/{song.name.lower().replace(' ', '-')}/"
                processed_song = web_extractor.process_song_with_web_url(song, generic_url)
                processed_songs.append(processed_song)
        
        # Mostrar resultados
        print(f"\n🎵 Análisis web completado:")
        for song in processed_songs:
            if song.lyrics_analysis:
                print(f"\n🎤 {song.name}:")
                print(f"   Sentimiento: {song.lyrics_analysis.sentiment.value}")
                print(f"   Visualizaciones: {song.lyrics_analysis.views:,}")
                print(f"   Compositores: {', '.join(song.lyrics_analysis.composers)}")
                print(f"   Temas: {song.lyrics_analysis.themes}")
                print(f"   Fuente: {song.lyrics_analysis.web_source}")
                print(f"   Letras (primeras 100 chars): {song.lyrics_analysis.lyrics[:100]}...")
        
        print("✅ Pruebas de extracción web completadas")
        return processed_songs
        
    except Exception as e:
        print(f"❌ Error en pruebas de extracción web: {e}")
        return []

def test_vector_database_with_web():
    """Función de prueba de base de datos vectorial con datos web"""
    print("🧪 Iniciando pruebas de base de datos vectorial con datos web...")
    
    try:
        # Verificar dependencias
        try:
            import chromadb
        except ImportError:
            print("⚠️ ChromaDB no instalado, saltando pruebas")
            return
        
        # Inicializar base de datos vectorial
        vector_db = VectorMusicDatabase()
        vector_db.initialize_database()
        
        # Obtener canciones procesadas con datos web
        processed_songs = test_web_lyrics_extraction()
        
        if processed_songs:
            # Guardar en base de datos vectorial
            vector_db.save_songs_to_vector_db(processed_songs)
            
            # Probar búsquedas específicas
            print("\n🔍 Probando búsquedas con datos web:")
            
            # Buscar por visualizaciones altas
            popular_songs = vector_db.search_songs_by_views(min_views=100000)
            print(f"Canciones populares (>100k views): {len(popular_songs.get('ids', [[]])[0])}")
            
            # Mostrar estadísticas completas con datos web
            stats = vector_db.get_statistics()
            print(f"\n📊 Estadísticas con datos web:")
            for key, value in stats.items():
                print(f"   {key}: {value}")
        
        print("✅ Pruebas de base de datos vectorial con datos web completadas")
        
    except Exception as e:
        print(f"❌ Error en pruebas de base de datos vectorial: {e}")

def main():
    """Función principal de demostración"""
    print("🎵 Solución Funcional: Sistema de Parseo Musical Radiohead con Web Scraping")
    print("=" * 70)
    
    try:
        # Prueba básica de parseo
        parser = RadioheadParser()
        parser.load_json()
        artist = parser.extract_artist_info()
        albums = parser.extract_albums()
        songs = parser.extract_songs()
        
        print(f"\n📊 Resultados básicos:")
        print(f"🎤 Artista: {artist.name}")
        print(f"🎭 Género: {artist.genre}")
        print(f"💿 Álbumes: {len(albums)}")
        print(f"🎵 Canciones: {len(songs)}")
        
        # Mostrar algunas canciones con sus URLs potenciales
        print(f"\n🎵 Canciones encontradas (con URLs de letras.com):")
        for song in songs[:5]:
            letras_url = f"https://www.letras.com/radiohead/{song.name.lower().replace(' ', '-').replace('(', '').replace(')', '')}/"
            print(f"  - {song.name} ({song.duration_formatted}) - {song.album_name}")
            print(f"    📄 URL: {letras_url}")
        
        print("\n✅ Parseo básico completado exitosamente")
        
        # Verificar si se puede hacer extracción web
        if os.getenv("OPENAI_API_KEY"):
            print("\n🌐 OPENAI_API_KEY configurada - Funcionalidad web disponible")
            print("💡 Ejecutando pruebas de extracción web...")
            
            # Ejecutar pruebas web
            # test_web_lyrics_extraction()
            # test_vector_database_with_web()
            
            test_urls = {
            "Let Down": "https://www.letras.com/radiohead/79031/",
            "Creep": "https://www.letras.com/radiohead/78989/",
            "Karma Police": "https://www.letras.com/radiohead/79030/"
            }
        
            # Inicializar extractor web
            web_extractor = WebLyricsExtractor()
            
            # Parsear canciones básicas
            parser = RadioheadParser()
            parser.load_json()
            songs = parser.extract_songs()
            
            # Procesar algunas canciones con URLs web
            processed_songs = []
            
            for song in songs[:5]:
                letras_url = f"https://www.letras.com/radiohead/{song.name.lower().replace(' ', '-').replace('(', '').replace(')', '')}/"
                url = letras_url
                processed_song = web_extractor.process_song_with_web_url(song, url)
                processed_songs.append(processed_song)
            
            print(f"\n🎵 Análisis web completado:")
            for song in processed_songs:
                if song.lyrics_analysis:
                    print(f"\n🎤 {song.name}:")
                    print(f"   Sentimiento: {song.lyrics_analysis.sentiment.value}")
                    print(f"   Visualizaciones: {song.lyrics_analysis.views:,}")
                    print(f"   Compositores: {', '.join(song.lyrics_analysis.composers)}")
                    print(f"   Temas: {song.lyrics_analysis.themes}")
                    print(f"   Fuente: {song.lyrics_analysis.web_source}")
                    print(f"   Letras (primeras 100 chars): {song.lyrics_analysis.lyrics[:100]}...")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Asegúrate de que el archivo data/radiohead.json existe")

if __name__ == "__main__":
    main()