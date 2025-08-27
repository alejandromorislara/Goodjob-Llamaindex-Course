# src/apis/news_api.py
from __future__ import annotations
import os
import argparse
from typing import List, Optional, Literal
import requests
from pydantic import BaseModel, HttpUrl, Field
from datetime import datetime
from llama_index.core.tools import FunctionTool

from dotenv import load_dotenv
load_dotenv()

NEWS_ENDPOINT = "https://newsapi.org/v2/everything"
DEFAULT_TIMEOUT = 10


# ---------------------------
# Pydantic models
# ---------------------------
class NewsInput(BaseModel):
    query: str = Field(..., description="Palabra(s) clave a buscar en titulares y texto")
    language: str = Field("es", description="Código de idioma (ej. 'es', 'en')")
    page_size: int = Field(10, description="Número máximo de artículos (1-100)")
    sort_by: Literal["relevancy", "popularity", "publishedAt"] = "publishedAt"


class Article(BaseModel):
    title: str
    description: Optional[str] = None
    url: HttpUrl
    source: str
    published_at: datetime = Field(alias="publishedAt")

    class Config:
        populate_by_name = True


class NewsResult(BaseModel):
    total: int
    articles: List[Article]


# ---------------------------
# Core function 
# ---------------------------
def fetch_news(
    query: str,
    language: str = "es",
    page_size: int = 10,
    sort_by: Literal["relevancy", "popularity", "publishedAt"] = "publishedAt",
) -> NewsResult:
    """
    Consulta NewsAPI con la query dada y devuelve artículos normalizados.
    Requiere NEWS_API_KEY en el entorno.
    """
    api_key = os.getenv("NEWS_API_KEY")
    if not api_key:
        raise RuntimeError("Falta NEWS_API_KEY en variables de entorno.")

    params = {
        "q": query,
        "language": language,
        "pageSize": max(1, min(page_size, 100)),
        "sortBy": sort_by,
        "apiKey": api_key,
    }

    try:
        resp = requests.get(NEWS_ENDPOINT, params=params, timeout=DEFAULT_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        raise RuntimeError(f"Error llamando a NewsAPI: {e}") from e

    if data.get("status") != "ok":
        raise RuntimeError(f"NewsAPI devolvió error: {data}")

    raw_articles = data.get("articles", [])
    articles: List[Article] = []
    for a in raw_articles:
        try:
            art = Article(
                title=a.get("title") or "",
                description=a.get("description"),
                url=a.get("url"),
                source=(a.get("source") or {}).get("name") or "desconocido",
                publishedAt=a.get("publishedAt"),
            )
            articles.append(art)
        except Exception:
            continue  # ignora artículos malformados

    return NewsResult(total=len(articles), articles=articles)


# ---------------------------
# Tool para LlamaIndex
# ---------------------------
def news_search_tool() -> FunctionTool:
    NewsInput.model_rebuild()
    Article.model_rebuild()
    NewsResult.model_rebuild()

    return FunctionTool.from_defaults(
        fn=fetch_news,
        name="news_search",
        description=(
            "Busca noticias recientes usando palabras clave. "
            "Útil para obtener información actualizada sobre cualquier tema."
        ),
        fn_schema=NewsInput,   
        return_direct=False,
    )


# ---------------------------
# CLI rápido
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--language", default="es")
    parser.add_argument("--page_size", type=int, default=5)
    parser.add_argument("--sort_by", default="publishedAt",
                        choices=["relevancy", "popularity", "publishedAt"])
    args = parser.parse_args()

    result = fetch_news(args.query, args.language, args.page_size, args.sort_by)
    for i, a in enumerate(result.articles, 1):
        print(f"[{i}] {a.published_at:%Y-%m-%d} — {a.source} — {a.title}\n{a.url}\n")
