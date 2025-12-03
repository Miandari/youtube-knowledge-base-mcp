"""
HyDE Transformer Service.

Transforms user queries into hypothetical documents for improved retrieval.
"""
import logging

from .provider import HyDEProvider

logger = logging.getLogger(__name__)


class HyDETransformer:
    """
    Service for transforming queries using HyDE.

    Generates hypothetical transcript snippets that bridge the vocabulary gap
    between formal search terms and informal YouTube transcript language.
    """

    def __init__(
        self,
        provider: HyDEProvider,
        max_tokens: int = 150,
    ):
        """
        Initialize the HyDE transformer.

        Args:
            provider: The HyDE provider for generation
            max_tokens: Maximum tokens for hypothetical document
        """
        self.provider = provider
        self.max_tokens = max_tokens
        logger.info(f"Initialized HyDE transformer with {provider.model_name}")

    def transform(self, query: str) -> str:
        """
        Transform a query into a hypothetical document.

        Args:
            query: The original search query

        Returns:
            Hypothetical document text (or original query on failure)
        """
        if not query or len(query.strip()) < 3:
            return query

        hypothetical_doc = self.provider.generate_hypothetical_document(
            query=query,
            max_tokens=self.max_tokens,
        )

        if hypothetical_doc and hypothetical_doc != query:
            logger.debug(f"HyDE transform: '{query[:50]}' -> '{hypothetical_doc[:50]}...'")
            return hypothetical_doc

        logger.debug(f"HyDE fallback to original query: '{query[:50]}'")
        return query
