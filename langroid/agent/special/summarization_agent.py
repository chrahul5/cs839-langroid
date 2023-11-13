"""
This file defines the summarization agent for long text documents.
"""
from typing import List
from langroid.agent.base import Agent

from langroid.mytypes import DocMetaData, Document, Entity
from langroid.agent.chat_document import ChatDocMetaData, ChatDocument
from langroid.agent.special.doc_chat_agent import DocChatAgentConfig, DocChatAgent
from langroid.language_models.base import StreamingIfAllowed
from langroid.language_models.openai_gpt import OpenAIChatModel, OpenAIGPTConfig

import logging

logger = logging.getLogger(__name__)

class SummarizationAgent(DocChatAgent):
    def __init__(self,
        config: DocChatAgentConfig,
    ):
        super().__init__(config)
        self.config: DocChatAgentConfig = config
        self.original_docs: None | List[Document] = None
        self.original_docs_length = 0
        self.chunked_docs: None | List[Document] = None
        self.chunked_docs_clean: None | List[Document] = None
        self.response: None | Document = None
        if len(config.doc_paths) > 0:
            self.ingest()

    def summarize_chunked_docs(
        self,
        instruction: str = "Give a concise summary of the following text:",
    ) -> None | ChatDocument:
        """Summarize all docs"""
        print("LOG: Using langroid summarization routine!")
        if self.llm is None:
            raise ValueError("LLM not set")
        if self.chunked_docs_clean is None:
            logger.warning(
                """
                No docs to summarize! 
                """
            )
            return None
        summary = ""
        for doc in self.chunked_docs_clean:
            chunk_text = doc.content
            tot_tokens = self.parser.num_tokens(chunk_text)
            MAX_INPUT_TOKENS = (
                self.llm.completion_context_length()
                - self.config.llm.max_output_tokens
                - 100
            )
            if tot_tokens > MAX_INPUT_TOKENS:
                # truncate
                chunk_text = self.parser.tokenizer.decode(
                    self.parser.tokenizer.encode(chunk_text)[:MAX_INPUT_TOKENS]
                )
                logger.warning(
                    f"Summarizing after truncating text to {MAX_INPUT_TOKENS} tokens"
                )
            prompt = f"""
            {instruction}
            {chunk_text}
            """.strip()
            with StreamingIfAllowed(self.llm):
                chunk_summary = Agent.llm_response(self, prompt)
                summary += chunk_summary.content  # type: ignore

        return ChatDocument(
                content=summary,
                metadata=ChatDocMetaData(
                    source=Entity.SYSTEM,
                    sender=Entity.LLM,
                ),
            )
