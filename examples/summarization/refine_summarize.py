"""
Example of a Langroid DocChatAgent summarization using query-refine approach.
"""
import typer
from rich import print
import langroid as lr
from langroid.agent.special.doc_chat_agent import DocChatAgentConfig
from langroid.agent.special.summarization_agent import SummarizationAgent
from langroid.parsing.parser import ParsingConfig, PdfParsingConfig, Splitter

app = typer.Typer()

lr.utils.logging.setup_colored_logging()

def chat() -> None:
    print(
        """
        [blue]Welcome to the retrieval-augmented chatbot!
        Enter x or q to quit
        """
    )


    # Initialize the doc agent.
    config = DocChatAgentConfig(
        n_query_rephrases=0,
        doc_paths=['examples/summarization/paul_graham.txt'],
        summary_paths=['examples/summarization/paul_graham_summary.txt'],
        cross_encoder_reranking_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        hypothetical_answer=False,
        parsing=ParsingConfig(  # modify as needed
            splitter=Splitter.TOKENS,
            chunk_size=1000,  # aim for this many tokens per chunk
            overlap=100,  # overlap between chunks
            max_chunks=10_000,
            # aim to have at least this many chars per chunk when
            # truncating due to punctuation
            min_chunk_chars=200,
            discard_chunk_chars=5,  # discard chunks with fewer than this many chars
            n_similar_docs=3,
            # NOTE: PDF parsing is extremely challenging, each library has its own
            # strengths and weaknesses. Try one that works for your use case.
            pdf=PdfParsingConfig(
                # alternatives: "haystack", "unstructured", "pdfplumber", "fitz"
                library="pdfplumber",
            ),
        ),
        vecdb=lr.vector_store.QdrantDBConfig(
            collection_name="quick-start-chat-agent-docs",
            replace_collection=True,
        ),
    )
    agent = lr.agent.special.SummarizationAgent(config)

    # Ingest default documents.
    agent.ingest()
    # summary = agent.tree_summarize_docs()
    refine_summary = agent.refine_summary(["what about colour classes the author took at RISD,?", "Who are Idelle and Julian Weber?"])
    print(f"summary of the doc: {refine_summary.content}")


@app.command()
def main(
        debug: bool = typer.Option(False, "--debug", "-d", help="debug mode"),
        no_stream: bool = typer.Option(False, "--nostream", "-ns", help="no streaming"),
        nocache: bool = typer.Option(False, "--nocache", "-nc", help="don't use cache"),
) -> None:
    lr.utils.configuration.set_global(
        lr.utils.configuration.Settings(
            debug=debug,
            cache=not nocache,
            stream=not no_stream,
        )
    )
    chat()


if __name__ == "__main__":
    app()