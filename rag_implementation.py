"""
Simple implementation of RAG-based summarization.
"""
import typer
from rich import print
import langroid as lr
from langroid.agent.special.doc_chat_agent import DocChatAgentConfig
from langroid.agent.special.summarization_agent import SummarizationAgent
from langroid.parsing.parser import ParsingConfig, PdfParsingConfig, Splitter

app = typer.Typer()

def summary(input_file) -> None:
    config = DocChatAgentConfig(
        n_query_rephrases=0,
        doc_paths=[input_file],
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

def chat(input_file, queries) -> None:
    print(
        """
        [blue]Welcome to the retrieval-augmented chatbot!
        Enter x or q to quit
        """
    )

#need to make changes to this config for relevant chunks retrieval purpose
    config = DocChatAgentConfig(
        n_query_rephrases=0,
        doc_paths=[input_file],
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
    chunk_dump = {}
    file_path = "chunk_dump.txt"

    # Retrieve relevant chunks for the given queries.
    doc_chat_agent = lr.agent.special.DocChatAgent(config)
    doc_chat_agent.ingest()
    for q in queries :
        query_chunks = doc_chat_agent.get_relevant_chunks(query=q)
        print(f"retrieved {len(query_chunks)} chunks")
        chunk_dump[q] = []
        chunk_dump[q].append(query_chunks[0].content)
        # for qc in query_chunks : 
        #     chunk_dump[q].append(qc.content)

    chunk_str = str(chunk_dump)
    with open(file_path, "w") as file:
        file.write(chunk_str)
    
    summary("chunk_dump.txt")

@app.command()
def main(
        debug: bool = typer.Option(False, "--debug", "-d", help="debug mode"),
        no_stream: bool = typer.Option(False, "--nostream", "-ns", help="no streaming"),
        nocache: bool = typer.Option(False, "--nocache", "-nc", help="don't use cache"),
) -> None:
    lr.utils.configuration.set_global(
        lr.utils.configuration.Settings(
            debug=debug,
            cache=nocache,
            stream=not no_stream,
        )
    )
    queries = ["Who are Idelle and Julian Weber?"]
    chat("examples/summarization/paul_graham.txt", queries=queries)


if __name__ == "__main__":
    app()