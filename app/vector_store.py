"""Qdrant persistence and document-filtered cosine retrieval."""

from qdrant_client import QdrantClient, models

from app.config import Settings
from app.schemas import Source


class VectorStore:
    def __init__(self, client: QdrantClient, settings: Settings) -> None:
        self.client = client
        self.settings = settings
        self.collection = settings.qdrant_collection

    def initialize(self) -> None:
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                self.collection,
                vectors_config=models.VectorParams(
                    size=self.settings.embedding_dimensions, distance=models.Distance.COSINE
                ),
            )
        info = self.client.get_collection(self.collection)
        vectors = info.config.params.vectors
        if not isinstance(vectors, models.VectorParams):
            raise ValueError("Expected one unnamed vector per Qdrant point")
        if (
            vectors.size != self.settings.embedding_dimensions
            or vectors.distance != models.Distance.COSINE
        ):
            raise ValueError("Qdrant collection configuration differs; use a new collection")

    def insert(
        self,
        document_id: str,
        filename: str,
        chunk_ids: list[str],
        chunks: list[str],
        embeddings: list[list[float]],
    ) -> None:
        if len(embeddings) != len(chunks):
            raise ValueError("Embedding count does not match chunk count")
        for start in range(0, len(chunks), 64):
            points = [
                models.PointStruct(
                    id=chunk_ids[i],
                    vector=embeddings[i],
                    payload={
                        "document_id": document_id,
                        "filename": filename,
                        "chunk_index": i,
                        "text": chunks[i],
                    },
                )
                for i in range(start, min(start + 64, len(chunks)))
            ]
            self.client.upsert(self.collection, points=points, wait=True)

    def delete_document(self, document_id: str) -> None:
        self.client.delete(
            self.collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_id", match=models.MatchValue(value=document_id)
                        )
                    ]
                )
            ),
            wait=True,
        )

    def search(self, embedding: list[float], document_ids: list[str]) -> list[Source]:
        if not document_ids:
            return []
        result = self.client.query_points(
            self.collection,
            query=embedding,
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_id", match=models.MatchAny(any=document_ids)
                    )
                ]
            ),
            limit=self.settings.top_k,
            score_threshold=self.settings.score_threshold,
            with_payload=True,
        )
        sources = []
        for point in result.points:
            payload = point.payload or {}
            sources.append(
                Source(
                    document_id=str(payload["document_id"]),
                    filename=str(payload["filename"]),
                    chunk_index=int(payload["chunk_index"]),
                    text=str(payload["text"]),
                    score=point.score,
                )
            )
        return sources
