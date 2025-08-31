from llama_index.core.embeddings import BaseEmbedding
from huggingface_hub import InferenceClient
from typing import Any

class HuggingFaceInferenceEmbedding(BaseEmbedding):
    # Usamos __init__ para inicializar los atributos necesarios
    def __init__(self, model: str, provider: str = "nebius"):
        super().__init__()
        # Asignamos los atributos usando __dict__ para evitar conflictos con Pydantic
        self.__dict__["client"] = InferenceClient(provider=provider)
        self.__dict__["model"] = model

    def _get_query_embedding(self, text: str):
        # Obtenemos la incrustación para un solo texto
        result = self.client.feature_extraction(
            text,
            model=self.model,
        )
        return result

    def _get_text_embedding(self, texts):
        # Obtenemos las incrustaciones para una lista de textos
        return [self._get_query_embedding(t) for t in texts]

    # Versión asíncrona: devuelve la incrustación para una sola cadena de consulta
    async def _aget_query_embedding(self, query: str):
        # No hay soporte async en InferenceClient, así que ejecutamos en un thread executor
        from concurrent.futures import ThreadPoolExecutor
        import asyncio
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as pool:
            # Ejecutamos _get_query_embedding en el thread pool
            result = await loop.run_in_executor(pool, self._get_query_embedding, query)
        return result
