"""
🎵 Template: Parseo Avanzado de Documentos Musicales con LlamaIndex y Persistencia

INSTRUCCIONES:
1. Completa las clases marcadas con # TODO
2. Configura las variables de entorno en .env
3. Ejecuta y prueba los casos de parseo y persistencia

ESTUDIANTE: ___________________
FECHA: _______________________
"""

import os
import sys
import json
import re
from datetime import datetime
from typing import List, Dict, Optional, Union, Any
from pathlib import Path

# Pydantic imports
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic import ValidationError
from dotenv import load_dotenv
from enum import Enum

# Agregar src al path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

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
# MODELOS PYDANTIC (6 puntos total)
# =============================================================================

class WebLyricsData(BaseModel):
    """TODO: Modelo para datos extraídos de la web con SimpleWebPageReader (1 punto)"""
    
    lyrics: str = Field(..., description="Letras completas extraídas de la página web")
    views: int = Field(default=0, ge=0, description="Número de visualizaciones")
    composers: List[str] = Field(default_factory=list, description="Compositores de la canción")
    additional_info: Optional[str] = Field(None, description="Información adicional relevante")
    source_url: str = Field(..., description="URL de origen de los datos")

class LyricsAnalysis(BaseModel):
    """TODO: Modelo para análisis de letras con LLM (1.5 puntos)"""
    
    lyrics: str = Field(..., description="Letras completas de la canción")
    sentiment: SentimentType = Field(..., description="Sentimiento principal de la canción")
    language: LanguageType = Field(..., description="Idioma detectado de las letras")
    themes: List[str] = Field(default_factory=list, description="Temas principales (máximo 5)")
    emotional_intensity: float = Field(..., ge=0.0, le=1.0, description="Intensidad emocional (0-1)")
    word_count: int = Field(..., ge=0, description="Número de palabras en las letras")
    views: int = Field(default=0, ge=0, description="Número de visualizaciones de la página web")
    composers: List[str] = Field(default_factory=list, description="Compositores de la canción")
    web_source: Optional[str] = Field(None, description="URL de origen de los datos")
    
    @field_validator('themes')
    @classmethod
    def validate_themes(cls, v):
        """TODO: Validar que no haya más de 5 temas"""
        # TODO: Implementar validación
        # if len(v) > 5:
        #     raise ValueError("Máximo 5 temas permitidos")
        # return [theme.strip().lower() for theme in v if theme.strip()]
        return v
    
    @field_validator('lyrics')
    @classmethod
    def validate_lyrics(cls, v):
        """TODO: Validar que las letras no estén vacías"""
        # TODO: Implementar validación
        # if not v or not v.strip():
        #     raise ValueError("Las letras no pueden estar vacías")
        # return v.strip()
        return v
    
    @model_validator(mode='after')
    def calculate_word_count(self):
        """TODO: Calcular automáticamente el número de palabras"""
        # TODO: Implementar cálculo
        # if self.lyrics:
        #     self.word_count = len(self.lyrics.split())
        return self

class Artist(BaseModel):
    """TODO: Modelo Pydantic para artista (1 punto)"""
    
    name: str = Field(..., description="Nombre del artista")
    genre: str = Field(..., description="Género musical principal")
    description: Optional[str] = Field(None, description="Descripción del artista")
    image_url: Optional[str] = Field(None, description="URL de la imagen del artista")
    url: Optional[str] = Field(None, description="URL oficial del artista")
    
    @field_validator('genre')
    @classmethod
    def validate_genre(cls, v):
        """TODO: Validar que el género esté en la lista permitida"""
        allowed_genres = ["Rock Alternativo", "Rock", "Electronic", "Experimental"]
        # TODO: Implementar validación
        # if v not in allowed_genres:
        #     raise ValueError(f"Género debe ser uno de: {allowed_genres}")
        return v
    
    @field_validator('description')
    @classmethod
    def sanitize_description(cls, v):
        """TODO: Sanitizar descripción eliminando caracteres especiales"""
        if v:
            # TODO: Implementar sanitización
            # return re.sub(r'[^\w\s\-.,]', '', v)
            pass
        return v
    
    @field_validator('name')
    @classmethod
    def normalize_name(cls, v):
        """TODO: Normalizar nombre a Title Case"""
        # TODO: return v.title() if v else v
        return v

class Album(BaseModel):
    """TODO: Modelo Pydantic para álbumes (1.5 puntos)"""
    
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
        """TODO: Validar que la fecha esté entre 1900-2030"""
        # TODO: Implementar validación de rango de fechas
        # if not (1900 <= v <= 2030):
        #     raise ValueError("Fecha de publicación debe estar entre 1900-2030")
        return v
    
    @field_validator('num_tracks')
    @classmethod
    def validate_num_tracks(cls, v):
        """TODO: Validar que el número de tracks esté entre 1-50"""
        # TODO: Implementar validación
        # if not (1 <= v <= 50):
        #     raise ValueError("Número de tracks debe estar entre 1-50")
        return v
    
    @field_validator('name')
    @classmethod
    def normalize_name(cls, v):
        """TODO: Normalizar nombre a Title Case"""
        # TODO: return v.title() if v else v
        return v
    
    @field_validator('publisher')
    @classmethod
    def validate_publisher(cls, v):
        """TODO: Validar que la editorial no esté vacía"""
        # TODO: Implementar validación
        # if not v or not v.strip():
        #     raise ValueError("Editorial no puede estar vacía")
        return v

class Song(BaseModel):
    """TODO: Modelo Pydantic para canciones con análisis de letras (2 puntos)"""
    
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
        """TODO: Validar formato ISO 8601 (PT3M55S)"""
        # TODO: Implementar validación con regex
        # pattern = r'^PT\d+M\d+S$'
        # if not re.match(pattern, v):
        #     raise ValueError("Duración debe estar en formato ISO 8601 (PT3M55S)")
        return v
    
    @model_validator(mode='after')
    def format_duration(self):
        """TODO: Convertir duración ISO a formato MM:SS"""
        # TODO: Implementar conversión
        # if self.duration_iso and not self.duration_formatted:
        #     duration_iso = self.duration_iso
        #     if duration_iso.startswith('PT') and 'M' in duration_iso and 'S' in duration_iso:
        #         # Extraer minutos y segundos
        #         minutes = int(duration_iso.split('M')[0][2:])
        #         seconds = int(duration_iso.split('M')[1].replace('S', ''))
        #         self.duration_formatted = f"{minutes}:{seconds:02d}"
        return self
    
    @field_validator('name')
    @classmethod
    def sanitize_name(cls, v):
        """TODO: Sanitizar nombre de la canción"""
        # TODO: Implementar sanitización
        return v

# =============================================================================
# CLASE PARSER PRINCIPAL (2 puntos)
# =============================================================================

class RadioheadParser:
    """TODO: Parser principal del documento JSON (2 puntos)"""
    
    def __init__(self, file_path: str = "data/radiohead.json"):
        self.file_path = file_path
        self.raw_data: Optional[Dict] = None
        self.artist: Optional[Artist] = None
        self.albums: List[Album] = []
        self.songs: List[Song] = []
    
    def load_json(self) -> Dict:
        """TODO: Cargar y validar estructura del JSON"""
        print(f"📁 Cargando {self.file_path}...")
        
        try:
            # TODO: Implementar carga del archivo JSON
            # with open(self.file_path, 'r', encoding='utf-8') as file:
            #     self.raw_data = json.load(file)
            
            # TODO: Validar estructura básica
            # required_keys = ['@context', '@type', 'name', 'track']
            # for key in required_keys:
            #     if key not in self.raw_data:
            #         raise ValueError(f"Clave requerida '{key}' no encontrada en JSON")
            
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
        """TODO: Extraer información del artista"""
        if not self.raw_data:
            raise ValueError("Debe cargar el JSON primero")
        
        print("🎤 Extrayendo información del artista...")
        
        # TODO: Implementar extracción de datos del artista
        # artist_data = {
        #     'name': self.raw_data.get('name', ''),
        #     'genre': self.raw_data.get('genre', ''),
        #     'description': self.raw_data.get('description', ''),
        #     'image_url': self.raw_data.get('image', ''),
        #     'url': self.raw_data.get('url', '')
        # }
        
        # TODO: Crear y validar modelo Artist
        # self.artist = Artist(**artist_data)
        
        print(f"✅ Artista extraído: {self.artist.name if self.artist else 'N/A'}")
        return self.artist
    
    def extract_albums(self) -> List[Album]:
        """TODO: Extraer álbumes únicos con metadatos"""
        if not self.raw_data:
            raise ValueError("Debe cargar el JSON primero")
        
        print("💿 Extrayendo álbumes...")
        
        albums_dict = {}
        
        # TODO: Implementar extracción de álbumes únicos
        # for track in self.raw_data.get('track', []):
        #     album_info = track.get('inAlbum', {})
        #     album_id = album_info.get('@id', '')
        #     
        #     if album_id and album_id not in albums_dict:
        #         album_data = {
        #             'id': album_id,
        #             'name': album_info.get('name', ''),
        #             'date_published': album_info.get('datePublished', 0),
        #             'description': album_info.get('description', ''),
        #             'genre': album_info.get('genre', ''),
        #             'image_url': album_info.get('image', ''),
        #             'num_tracks': album_info.get('numtracks', 0),
        #             'publisher': album_info.get('publisher', ''),
        #             'url': album_info.get('url', '')
        #         }
        #         
        #         try:
        #             albums_dict[album_id] = Album(**album_data)
        #         except ValidationError as e:
        #             print(f"⚠️ Error validando álbum {album_data.get('name', 'N/A')}: {e}")
        
        self.albums = list(albums_dict.values())
        print(f"✅ {len(self.albums)} álbumes extraídos")
        return self.albums
    
    def extract_songs(self) -> List[Song]:
        """TODO: Extraer canciones con información completa"""
        if not self.raw_data:
            raise ValueError("Debe cargar el JSON primero")
        
        print("🎵 Extrayendo canciones...")
        
        songs = []
        
        # TODO: Implementar extracción de canciones
        # for track in self.raw_data.get('track', []):
        #     song_data = {
        #         'id': track.get('@id', ''),
        #         'name': track.get('name', ''),
        #         'description': track.get('description', ''),
        #         'duration_iso': track.get('duration', ''),
        #         'genre': track.get('genre', ''),
        #         'image_url': track.get('image', ''),
        #         'url': track.get('url', ''),
        #         'album_name': track.get('inAlbum', {}).get('name', ''),
        #         'album_id': track.get('inAlbum', {}).get('@id', ''),
        #         'artist_name': track.get('byArtist', {}).get('name', ''),
        #         'artist_id': track.get('byArtist', {}).get('@id', '')
        #     }
        #     
        #     try:
        #         songs.append(Song(**song_data))
        #     except ValidationError as e:
        #         print(f"⚠️ Error validando canción {song_data.get('name', 'N/A')}: {e}")
        
        self.songs = songs
        print(f"✅ {len(self.songs)} canciones extraídas")
        return self.songs


# =============================================================================
# CLASE EXTRACTOR DE LETRAS CON LLM (2.5 puntos)
# =============================================================================

class WebLyricsExtractor:
    """TODO: Extractor de letras desde páginas web usando SimpleWebPageReader (3 puntos)"""
    
    def __init__(self):
        # Configurar LlamaIndex
        Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0.0)
        Settings.embed_model = OpenAIEmbedding()
        self.llm = Settings.llm
        self.web_reader = SimpleWebPageReader()
    
    def extract_lyrics_from_url(self, url: str, song_name: str, artist_name: str) -> Optional[WebLyricsData]:
        """TODO: Extraer letras y datos desde una URL usando SimpleWebPageReader y LLM tipado"""
        print(f"🌐 Extrayendo datos de: {url}")
        
        try:
            # TODO: Leer contenido de la página web
            # documents = self.web_reader.load_data([url])
            # 
            # if not documents:
            #     print(f"❌ No se pudo cargar contenido de {url}")
            #     return None
            # 
            # # Obtener el contenido de la página
            # page_content = documents[0].text
            # 
            # # Crear prompt para extraer datos estructurados
            # prompt_template = """
            # Extrae información de esta página web de letras de canciones y devuélvela en el formato JSON especificado.
            # 
            # Contenido de la página:
            # {page_content}
            # 
            # Canción: {song_name}
            # Artista: {artist_name}
            # URL: {url}
            # 
            # Extrae la siguiente información y devuélvela como un objeto JSON válido:
            # - lyrics: Las letras completas de la canción (texto limpio, sin HTML)
            # - views: El número de visualizaciones (busca números como "290,298 visualizaciones" o similares)
            # - composers: Lista de compositores (busca "Compuesta por:" o similar)
            # - additional_info: Cualquier información adicional relevante
            # - source_url: La URL proporcionada ({url})
            # 
            # IMPORTANTE: Devuelve ÚNICAMENTE el objeto JSON, sin texto adicional, sin markdown, sin explicaciones.
            # """
            # 
            # # Usar LLM con output tipado
            # program = LLMTextCompletionProgram.from_defaults(
            #     output_cls=WebLyricsData,
            #     prompt_template_str=prompt_template,
            #     llm=self.llm
            # )
            # 
            # result = program(
            #     page_content=page_content[:1000],  # Limitar contenido para evitar tokens excesivos
            #     song_name=song_name,
            #     artist_name=artist_name,
            #     url=url
            # )
            # 
            # print(f"✅ Datos extraídos de {url}")
            # return result
            
            # Placeholder para implementación
            return None
            
        except Exception as e:
            print(f"❌ Error extrayendo datos de {url}: {e}")
            return None
    
    def analyze_lyrics_with_web_data(self, web_data: WebLyricsData, song_name: str, artist_name: str) -> Optional[LyricsAnalysis]:
        """TODO: Analizar letras extraídas de la web y crear análisis completo"""
        print(f"🎤 Analizando letras de: {song_name} - {artist_name}")
        
        try:
            # TODO: Crear prompt para análisis de sentimiento y temas
            # prompt_template = """
            # Analiza las letras de la canción "{song_name}" de {artist_name} y proporciona un análisis estructurado.
            # 
            # Letras de la canción:
            # {lyrics}
            # 
            # Información adicional:
            # - Visualizaciones: {views}
            # - Compositores: {composers}
            # 
            # Proporciona un análisis completo en formato JSON con los siguientes campos:
            # - lyrics: Las letras completas (copia exacta del input)
            # - sentiment: Uno de estos valores exactos: "positive", "negative", "neutral", "melancholic", "energetic", "romantic", "angry"
            # - language: Código de idioma: "en", "es", "fr", "it", "unknown"
            # - themes: Lista de máximo 5 temas principales (como strings)
            # - emotional_intensity: Número decimal entre 0.0 y 1.0
            # - word_count: Número entero de palabras en las letras
            # - views: El número de visualizaciones (usar el valor proporcionado: {views})
            # - composers: Lista de compositores (usar la lista proporcionada)
            # - web_source: URL de origen (dejar como null por ahora)
            # 
            # IMPORTANTE: Devuelve ÚNICAMENTE el objeto JSON válido, sin markdown, sin explicaciones adicionales.
            # """
            # 
            # # Usar LLM con output tipado
            # program = LLMTextCompletionProgram.from_defaults(
            #     output_cls=LyricsAnalysis,
            #     prompt_template_str=prompt_template,
            #     llm=self.llm
            # )
            # 
            # result = program(
            #     song_name=song_name,
            #     artist_name=artist_name,
            #     lyrics=web_data.lyrics,
            #     views=web_data.views,
            #     composers=", ".join(web_data.composers) if web_data.composers else "N/A"
            # )
            # 
            # # Agregar datos web al análisis
            # result.views = web_data.views
            # result.composers = web_data.composers
            # result.web_source = web_data.source_url
            # 
            # print(f"✅ Análisis completado para {song_name}")
            # return result
            
            # Placeholder para implementación
            return None
            
        except Exception as e:
            print(f"❌ Error analizando letras de {song_name}: {e}")
            return None
    
    def process_song_with_web_url(self, song: Song, letras_url: str) -> Song:
        """TODO: Procesar una canción completa con su URL de letras.com"""
        print(f"🎵 Procesando {song.name} con URL: {letras_url}")
        
        # TODO: Extraer datos de la web
        # web_data = self.extract_lyrics_from_url(letras_url, song.name, song.artist_name)
        # 
        # if web_data:
        #     # Analizar letras
        #     lyrics_analysis = self.analyze_lyrics_with_web_data(web_data, song.name, song.artist_name)
        #     
        #     if lyrics_analysis:
        #         song.lyrics_analysis = lyrics_analysis
        #         print(f"✅ {song.name} procesada exitosamente")
        #     else:
        #         print(f"⚠️ No se pudo analizar las letras de {song.name}")
        # else:
        #     print(f"⚠️ No se pudieron extraer datos web de {song.name}")
        
        return song
    
    def batch_extract_lyrics_from_web(self, songs: List[Song]) -> List[Song]:
        """TODO: Extraer letras para múltiples canciones desde web en lote"""
        print(f"🌐 Extrayendo letras web para {len(songs)} canciones...")
        
        updated_songs = []
        
        for i, song in enumerate(songs, 1):
            print(f"📝 Procesando {i}/{len(songs)}: {song.name}")
            
            # Generar URL de letras.com
            letras_url = f"https://www.letras.com/radiohead/{song.name.lower().replace(' ', '-').replace('(', '').replace(')', '').replace('/', '').replace('&', 'and')}/"
            
            # Procesar canción
            processed_song = self.process_song_with_web_url(song, letras_url)
            updated_songs.append(processed_song)
        
        print(f"✅ Procesamiento web completado")
        return updated_songs

class LyricsExtractor:
    """TODO: Extractor de letras y sentimiento con LLM tipado (2.5 puntos) - DEPRECADO - Usar WebLyricsExtractor"""
    
    def __init__(self):
        # Configurar LlamaIndex
        Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0.0)
        self.llm = Settings.llm
    
    def extract_lyrics_and_sentiment(self, song_name: str, artist_name: str, album_name: str) -> Optional[LyricsAnalysis]:
        """TODO: Extraer letras y análisis de sentimiento usando LLM tipado"""
        print(f"🎤 Extrayendo letras para: {song_name} - {artist_name}")
        
        # TODO: Crear prompt para extraer letras
        # prompt = f"""
        # Analiza la canción "{song_name}" de {artist_name} del álbum "{album_name}".
        # 
        # Proporciona un análisis completo que incluya:
        # 1. Las letras completas de la canción (si las conoces)
        # 2. El sentimiento principal
        # 3. El idioma de las letras
        # 4. Los temas principales (máximo 5)
        # 5. La intensidad emocional (0.0 a 1.0)
        # 
        # Si no conoces las letras exactas, proporciona letras representativas basadas en el estilo de la banda.
        # """
        
        try:
            # TODO: Usar LLM con output tipado
            # from llama_index.core.program import LLMTextCompletionProgram
            # program = LLMTextCompletionProgram.from_defaults(
            #     output_cls=LyricsAnalysis,
            #     prompt_template_str=prompt,
            #     llm=self.llm
            # )
            # 
            # result = program()
            # return result
            
            # Placeholder para implementación
            return None
            
        except Exception as e:
            print(f"❌ Error extrayendo letras para {song_name}: {e}")
            return None
    
    def batch_extract_lyrics(self, songs: List[Song]) -> List[Song]:
        """TODO: Extraer letras para múltiples canciones en lote"""
        print(f"🎵 Extrayendo letras para {len(songs)} canciones...")
        
        updated_songs = []
        
        for i, song in enumerate(songs, 1):
            print(f"📝 Procesando {i}/{len(songs)}: {song.name}")
            
            # TODO: Extraer letras y sentimiento
            # lyrics_analysis = self.extract_lyrics_and_sentiment(
            #     song.name, 
            #     song.artist_name, 
            #     song.album_name
            # )
            
            # TODO: Actualizar canción con análisis
            # if lyrics_analysis:
            #     song.lyrics_analysis = lyrics_analysis
            
            updated_songs.append(song)
        
        print(f"✅ Procesamiento de letras completado")
        return updated_songs

# =============================================================================
# CLASE BASE DE DATOS VECTORIAL (3 puntos)
# =============================================================================

class VectorMusicDatabase:
    """TODO: Base de datos vectorial persistente para música (4 puntos)"""
    
    def __init__(self, persist_dir: str = "./chroma_music_db"):
        self.persist_dir = persist_dir
        self.collection_name = "radiohead_songs"
        self.client = None
        self.collection = None
        
        # Configurar LlamaIndex
        Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0.0)
        Settings.embed_model = OpenAIEmbedding()
    
    def initialize_database(self):
        """TODO: Inicializar base de datos vectorial persistente"""
        print("🗄️ Inicializando base de datos vectorial...")
        
        try:
            # TODO: Importar ChromaDB
            # import chromadb
            # from chromadb.config import Settings as ChromaSettings
            
            # TODO: Crear cliente persistente
            # self.client = chromadb.PersistentClient(
            #     path=self.persist_dir,
            #     settings=ChromaSettings(anonymized_telemetry=False)
            # )
            
            # TODO: Crear o obtener colección
            # self.collection = self.client.get_or_create_collection(
            #     name=self.collection_name,
            #     metadata={"description": "Radiohead songs with lyrics and sentiment analysis"}
            # )
            
            print("✅ Base de datos vectorial inicializada")
            
        except Exception as e:
            print(f"❌ Error inicializando base de datos: {e}")
            raise
    
    def save_songs_to_vector_db(self, songs: List[Song]):
        """TODO: Guardar canciones con letras en base de datos vectorial"""
        if not self.collection:
            raise ValueError("Debe inicializar la base de datos primero")
        
        print(f"💾 Guardando {len(songs)} canciones en base de datos vectorial...")
        
        documents = []
        metadatas = []
        ids = []
        
        for song in songs:
            if song.lyrics_analysis:
                # TODO: Crear documento con letras
                # doc_text = f"""
                # Canción: {song.name}
                # Artista: {song.artist_name}
                # Álbum: {song.album_name}
                # Género: {song.genre}
                # Duración: {song.duration_formatted}
                # 
                # Letras:
                # {song.lyrics_analysis.lyrics}
                # """
                
                # TODO: Crear metadata completa
                # metadata = {
                #     "song_id": song.id,
                #     "song_name": song.name,
                #     "artist_name": song.artist_name,
                #     "album_name": song.album_name,
                #     "genre": song.genre,
                #     "duration": song.duration_formatted,
                #     "sentiment": song.lyrics_analysis.sentiment.value,
                #     "language": song.lyrics_analysis.language.value,
                #     "emotional_intensity": song.lyrics_analysis.emotional_intensity,
                #     "word_count": song.lyrics_analysis.word_count,
                #     "themes": ",".join(song.lyrics_analysis.themes),
                #     "url": song.url
                # }
                
                # TODO: Agregar a listas
                # documents.append(doc_text)
                # metadatas.append(metadata)
                # ids.append(f"song_{song.id}")
                pass
        
        try:
            # TODO: Insertar en ChromaDB
            # if documents:
            #     self.collection.add(
            #         documents=documents,
            #         metadatas=metadatas,
            #         ids=ids
            #     )
            
            print(f"✅ {len(documents)} canciones guardadas en base de datos vectorial")
            
        except Exception as e:
            print(f"❌ Error guardando en base de datos vectorial: {e}")
            raise
    
    def search_songs_by_sentiment(self, sentiment: SentimentType, limit: int = 5) -> List[Dict]:
        """TODO: Buscar canciones por sentimiento"""
        if not self.collection:
            raise ValueError("Debe inicializar la base de datos primero")
        
        print(f"🔍 Buscando canciones con sentimiento: {sentiment.value}")
        
        try:
            # TODO: Buscar por metadata
            # results = self.collection.query(
            #     where={"sentiment": sentiment.value},
            #     n_results=limit
            # )
            # return results
            
            return []
            
        except Exception as e:
            print(f"❌ Error buscando por sentimiento: {e}")
            return []
    
    def search_songs_by_lyrics(self, query: str, limit: int = 5) -> List[Dict]:
        """TODO: Buscar canciones por contenido de letras"""
        if not self.collection:
            raise ValueError("Debe inicializar la base de datos primero")
        
        print(f"🔍 Buscando canciones por letras: {query}")
        
        try:
            # TODO: Búsqueda semántica
            # results = self.collection.query(
            #     query_texts=[query],
            #     n_results=limit
            # )
            # return results
            
            return []
            
        except Exception as e:
            print(f"❌ Error buscando por letras: {e}")
            return []
    
    def get_all_songs(self) -> List[Dict]:
        """TODO: Obtener todas las canciones de la base de datos vectorial"""
        if not self.collection:
            raise ValueError("Debe inicializar la base de datos primero")
        
        print("📋 Obteniendo todas las canciones...")
        
        try:
            # TODO: Obtener todas las canciones
            # results = self.collection.get()
            # return results
            
            return []
            
        except Exception as e:
            print(f"❌ Error obteniendo canciones: {e}")
            return []
    
    def search_songs_by_album(self, album_name: str, limit: int = 10) -> List[Dict]:
        """TODO: Buscar canciones por álbum"""
        if not self.collection:
            raise ValueError("Debe inicializar la base de datos primero")
        
        print(f"🔍 Buscando canciones del álbum: {album_name}")
        
        try:
            # TODO: Buscar por metadata de álbum
            # results = self.collection.query(
            #     where={"album_name": album_name},
            #     n_results=limit
            # )
            # return results
            
            return []
            
        except Exception as e:
            print(f"❌ Error buscando por álbum: {e}")
            return []
    
    def search_songs_by_themes(self, themes: List[str], limit: int = 5) -> List[Dict]:
        """TODO: Buscar canciones por temas específicos"""
        if not self.collection:
            raise ValueError("Debe inicializar la base de datos primero")
        
        print(f"🔍 Buscando canciones con temas: {themes}")
        
        try:
            # TODO: Buscar por temas en metadata
            # # Crear filtro para temas (buscar si algún tema está presente)
            # theme_filters = []
            # for theme in themes:
            #     theme_filters.append({"themes": {"$contains": theme}})
            # 
            # results = self.collection.query(
            #     where={"$or": theme_filters},
            #     n_results=limit
            # )
            # return results
            
            return []
            
        except Exception as e:
            print(f"❌ Error buscando por temas: {e}")
            return []
    
    def get_statistics(self) -> Dict[str, Any]:
        """TODO: Obtener estadísticas completas de la base de datos vectorial"""
        if not self.collection:
            raise ValueError("Debe inicializar la base de datos primero")
        
        print("📊 Obteniendo estadísticas de base de datos vectorial...")
        
        try:
            # TODO: Obtener estadísticas completas
            # count = self.collection.count()
            # 
            # # Estadísticas por sentimiento
            # sentiment_stats = {}
            # for sentiment in SentimentType:
            #     results = self.collection.query(
            #         where={"sentiment": sentiment.value},
            #         n_results=1000  # Obtener todas
            #     )
            #     sentiment_stats[sentiment.value] = len(results['ids'])
            # 
            # # Estadísticas por álbum
            # all_songs = self.collection.get()
            # albums = set()
            # languages = set()
            # total_intensity = 0
            # 
            # for metadata in all_songs['metadatas']:
            #     albums.add(metadata.get('album_name', 'Unknown'))
            #     languages.add(metadata.get('language', 'unknown'))
            #     total_intensity += float(metadata.get('emotional_intensity', 0))
            # 
            # avg_intensity = total_intensity / count if count > 0 else 0
            # 
            # return {
            #     "total_songs": count,
            #     "total_albums": len(albums),
            #     "languages_detected": list(languages),
            #     "sentiment_distribution": sentiment_stats,
            #     "average_emotional_intensity": round(avg_intensity, 2),
            #     "collection_name": self.collection_name,
            #     "persist_dir": self.persist_dir
            # }
            
            return {}
            
        except Exception as e:
            print(f"❌ Error obteniendo estadísticas: {e}")
            return {}
    
    def delete_all_songs(self):
        """TODO: Eliminar todas las canciones (útil para testing)"""
        if not self.collection:
            raise ValueError("Debe inicializar la base de datos primero")
        
        print("🗑️ Eliminando todas las canciones...")
        
        try:
            # TODO: Eliminar toda la colección y recrearla
            # self.client.delete_collection(self.collection_name)
            # self.collection = self.client.create_collection(
            #     name=self.collection_name,
            #     metadata={"description": "Radiohead songs with lyrics and sentiment analysis"}
            # )
            
            print("✅ Todas las canciones eliminadas")
            
        except Exception as e:
            print(f"❌ Error eliminando canciones: {e}")
            raise

# =============================================================================
# CLASE LLAMAINDEX PROCESSOR (2 puntos)
# =============================================================================

class LlamaIndexProcessor:
    """TODO: Procesador con LlamaIndex (2.5 puntos)"""
    
    def __init__(self):
        # Configurar LlamaIndex
        Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0.0)
        Settings.embed_model = OpenAIEmbedding()
        
        self.documents: List[Document] = []
        self.index: Optional[VectorStoreIndex] = None
        self.query_engine = None
    
    def create_documents(self, artist: Artist, albums: List[Album], songs: List[Song]) -> List[Document]:
        """TODO: Convertir datos a documentos LlamaIndex"""
        print("📄 Creando documentos LlamaIndex...")
        
        documents = []
        
        # TODO: Crear documento para el artista
        # TODO: Crear documentos para álbumes
        # TODO: Crear documentos para canciones
        
        self.documents = documents
        print(f"✅ {len(documents)} documentos creados")
        return documents
    
    def create_index(self) -> VectorStoreIndex:
        """TODO: Crear índice vectorial"""
        # TODO: Implementar método completo
        print("🔍 Creando índice vectorial...")
        return self.index
    
    def query_music_info(self, question: str) -> str:
        """TODO: Consultar información musical con LLM"""
        # TODO: Implementar método completo
        print(f"❓ Consultando: {question}")
        return "TODO: Implementar consulta"

# =============================================================================
# FUNCIONES DE PRUEBA Y MAIN
# =============================================================================

def test_parsing():
    """TODO: Función de prueba del parseo"""
    print("🧪 Iniciando pruebas de parseo...")
    
    try:
        # TODO: Inicializar parser
        # TODO: Cargar y parsear JSON
        # TODO: Mostrar resultados
        
        print("✅ Pruebas de parseo completadas")
        
    except Exception as e:
        print(f"❌ Error en pruebas de parseo: {e}")

def test_web_lyrics_extraction():
    """TODO: Función de prueba de extracción de letras desde web (2 puntos)"""
    print("🧪 Iniciando pruebas de extracción web...")
    
    try:
        # TODO: URLs de ejemplo de letras.com para Radiohead
        test_urls = {
            "Let Down": "https://www.letras.com/radiohead/79031/",
            "Creep": "https://www.letras.com/radiohead/78989/",
            "Karma Police": "https://www.letras.com/radiohead/79030/"
        }
        
        print("🌐 URLs que se procesarían:")
        for song, url in test_urls.items():
            print(f"  - {song}: {url}")
        
        # TODO: Inicializar extractor web
        # web_extractor = WebLyricsExtractor()
        
        # TODO: Parsear canciones básicas
        # parser = RadioheadParser()
        # parser.load_json()
        # songs = parser.extract_songs()
        
        # TODO: Procesar TODAS las canciones con URLs web
        # processed_songs = web_extractor.batch_extract_lyrics_from_web(songs)
        
        # TODO: Mostrar resultados con datos web
        # print(f"\n🎵 Análisis web completado:")
        # songs_with_analysis = [s for s in processed_songs if s.lyrics_analysis]
        # 
        # for song in songs_with_analysis:
        #     print(f"\n🎤 {song.name}:")
        #     print(f"   Sentimiento: {song.lyrics_analysis.sentiment.value}")
        #     print(f"   Visualizaciones: {song.lyrics_analysis.views:,}")
        #     print(f"   Compositores: {', '.join(song.lyrics_analysis.composers)}")
        #     print(f"   Temas: {song.lyrics_analysis.themes}")
        #     print(f"   Fuente: {song.lyrics_analysis.web_source}")
        
        print("✅ Pruebas de extracción web completadas")
        
    except Exception as e:
        print(f"❌ Error en pruebas de extracción web: {e}")

def test_lyrics_extraction():
    """TODO: Función de prueba de extracción de letras - REDIRIGIDA A WEB"""
    print("⚠️ Redirigiendo a extracción web...")
    return test_web_lyrics_extraction()

def test_vector_database():
    """TODO: Función de prueba de la base de datos vectorial"""
    print("🧪 Iniciando pruebas de base de datos vectorial...")
    
    try:
        # TODO: Inicializar base de datos vectorial
        # vector_db = VectorMusicDatabase()
        # vector_db.initialize_database()
        
        # TODO: Parsear y extraer letras
        # parser = RadioheadParser()
        # parser.load_json()
        # songs = parser.extract_songs()
        
        # extractor = LyricsExtractor()
        # songs_with_lyrics = extractor.batch_extract_lyrics(songs[:5])  # Solo 5 para prueba
        
        # TODO: Guardar en base de datos vectorial
        # vector_db.save_songs_to_vector_db(songs_with_lyrics)
        
        # TODO: Probar búsquedas avanzadas
        # print("\n🔍 Probando búsquedas:")
        # 
        # # Buscar por sentimiento
        # melancholic_songs = vector_db.search_songs_by_sentiment(SentimentType.MELANCHOLIC)
        # print(f"Canciones melancólicas encontradas: {len(melancholic_songs)}")
        # 
        # # Buscar por letras
        # love_songs = vector_db.search_songs_by_lyrics("love and relationships")
        # print(f"Canciones sobre amor encontradas: {len(love_songs)}")
        # 
        # # Buscar por álbum
        # ok_computer_songs = vector_db.search_songs_by_album("OK Computer")
        # print(f"Canciones de OK Computer encontradas: {len(ok_computer_songs)}")
        # 
        # # Buscar por temas
        # existential_songs = vector_db.search_songs_by_themes(["existentialism", "alienation"])
        # print(f"Canciones existenciales encontradas: {len(existential_songs)}")
        
        # TODO: Mostrar estadísticas completas
        # stats = vector_db.get_statistics()
        # print(f"\n📊 Estadísticas de la base de datos vectorial:")
        # for key, value in stats.items():
        #     print(f"   {key}: {value}")
        # 
        # # Mostrar todas las canciones (solo nombres)
        # all_songs = vector_db.get_all_songs()
        # if all_songs and 'metadatas' in all_songs:
        #     print(f"\n🎵 Canciones en la base de datos:")
        #     for metadata in all_songs['metadatas'][:5]:  # Solo las primeras 5
        #         print(f"   - {metadata.get('song_name', 'N/A')} ({metadata.get('sentiment', 'N/A')})")
        
        print("✅ Pruebas de base de datos vectorial completadas")
        
    except Exception as e:
        print(f"❌ Error en pruebas de base de datos vectorial: {e}")

def test_llamaindex():
    """TODO: Función de prueba de LlamaIndex"""
    print("🧪 Iniciando pruebas de LlamaIndex...")
    
    try:
        # TODO: Implementar pruebas completas
        print("✅ Pruebas de LlamaIndex completadas")
        
    except Exception as e:
        print(f"❌ Error en pruebas de LlamaIndex: {e}")

def main():
    """TODO: Función principal"""
    print("🎵 Sistema de Parseo Musical Radiohead con Análisis de Letras")
    print("=" * 60)
    
    # TODO: Verificar variables de entorno
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️ OPENAI_API_KEY no configurada")
        return
    
    # TODO: Ejecutar todas las pruebas
    print("\n1️⃣ Ejecutando pruebas de parseo...")
    test_parsing()
    
    print("\n2️⃣ Ejecutando pruebas de extracción web de letras...")
    test_web_lyrics_extraction()
    
    print("\n3️⃣ Ejecutando pruebas de base de datos vectorial...")
    test_vector_database()
    
    print("\n4️⃣ Ejecutando pruebas de LlamaIndex...")
    test_llamaindex()
    
    print("\n🎉 ¡Todas las pruebas completadas!")
    print("\n💡 El sistema ahora incluye:")
    print("   - Extracción de letras con LLM tipado")
    print("   - Análisis de sentimiento automático")
    print("   - Base de datos vectorial persistente")
    print("   - Búsquedas semánticas por letras y sentimiento")

if __name__ == "__main__":
    main()