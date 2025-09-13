import re
from urllib.parse import urljoin

import scrapy
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule
from w3lib.html import remove_tags, replace_escape_chars


def clean_text(html_fragment: str) -> str:
    # Limpieza básica: eliminar HTML y normalizar espacios
    txt = remove_tags(html_fragment or "")
    txt = replace_escape_chars(txt, which_ones=("&nbsp;",))
    txt = re.sub(r"[ \t]+", " ", txt)
    txt = re.sub(r"\n{2,}", "\n", txt).strip()
    return txt


class LlamaIndexDocsSpider(CrawlSpider):
    name = "docs_spider"
    allowed_domains = ["docs.llamaindex.ai"]
    start_urls = ["https://docs.llamaindex.ai/en/stable/"]

    # Solo seguimos enlaces que permanezcan bajo /en/stable/
    rules = (
        Rule(
            LinkExtractor(allow=(r"/en/stable/"), allow_domains=allowed_domains),
            callback="parse_page",
            follow=True,
        ),
    )

    custom_settings = {
        # Respeto a robots.txt
        "ROBOTSTXT_OBEY": True,
        # Autothrottle para no sobrecargar el sitio
        "AUTOTHROTTLE_ENABLED": True,
        "AUTOTHROTTLE_START_DELAY": 0.5,
        "AUTOTHROTTLE_MAX_DELAY": 5.0,
        "DOWNLOAD_DELAY": 0.25,
        # User-Agent identificable
        "DEFAULT_REQUEST_HEADERS": {
            "User-Agent": "llamaindex-docs-crawler (+https://example.org/educational-use)"
        },
        # Export directo a ficheros
        "FEEDS": {
            "outputs/llamaindex_docs.jsonl": {
                "format": "jsonlines",
                "overwrite": True,
                "encoding": "utf-8",
            },
            "outputs/llamaindex_docs.csv": {
                "format": "csv",
                "overwrite": True,
                "encoding": "utf-8",
            },
        },
        # Opcional: desactiva logs muy verbosos
        # "LOG_LEVEL": "INFO",
    }

    def parse_page(self, response):
        # Sphinx/pydata-theme suele tener el contenido dentro de un rol main
        # probamos varios selectores por robustez.
        content_selectors = [
            'main[role="main"]',
            "div.bd-main div.bd-content",     # PyData Sphinx Theme
            "div.document",                   # Fallback Sphinx
            "article",                        # Fallback genérico
        ]

        html_content = ""
        for sel in content_selectors:
            fragment = response.css(sel).get()
            if fragment and len(fragment) > 300:
                html_content = fragment
                break
        if not html_content:
            # Fallback: toda la página (menos eficiente, pero evita perder contenido)
            html_content = response.text

        text_content = clean_text(html_content)

        title = response.css("h1::text").get()
        if not title:
            # Fallback al <title> de la página
            title = (response.css("title::text").get() or "").strip()

        yield {
            "url": response.url,
            "title": title,
            "text": text_content,
            "metadata": {
                "source": "docs.llamaindex.ai",
                "lang": "en",
                "section": "stable",
            },
        }
