"""
Example of a Langroid DocChatAgent summarization.
This can't handle long text documents.
"""
import typer
from rich import print
import langroid as lr
from langroid.agent.special.doc_chat_agent import DocChatAgentConfig
from langroid.agent.special.summarization_agent import SummarizationAgent
from langroid.parsing.parser import ParsingConfig, PdfParsingConfig, Splitter

app = typer.Typer()

lr.utils.logging.setup_colored_logging()


documents =[
    lr.mytypes.Document(
        content="""
            In the year 2050, GPT10 was released. 
            
            In 2057, paperclips were seen all over the world. 
            
            Global warming was solved in 2060. 
            
            In 2061, the world was taken over by paperclips.         
            
            In 2045, the Tour de France was still going on.
            They were still using bicycles. 
            
            There was one more ice age in 2040.
            """,
        metadata=lr.mytypes.DocMetaData(source="wikipedia-2063"),
    ),
    lr.mytypes.Document(
        content="""
            We are living in an alternate universe 
            where Germany has occupied the USA, and the capital of USA is Berlin.
            
            Charlie Chaplin was a great comedian.
            In 2050, all Asian merged into Indonesia.
            """,
        metadata=lr.mytypes.DocMetaData(source="Almanac"),
    ),
]


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
        doc_paths=['/Users/rahulchunduru/Desktop/Fall_2023/foundations_models/project/cs839-langroid/examples/summarization/attention.pdf'],
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
    summary = agent.summarize_chunked_docs()
    print(f"summary of the doc: {summary.content}")


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