"""
Example of a Langroid DocChatAgent summarization.
This can't handle long text documents.
"""
import typer
from rich import print
import langroid as lr

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


    config = lr.agent.special.DocChatAgentConfig(
        llm = lr.language_models.OpenAIGPTConfig(
            chat_model=lr.language_models.OpenAIChatModel.GPT4,
        ),
        vecdb=lr.vector_store.QdrantDBConfig(
            collection_name="quick-start-chat-agent-docs",
            replace_collection=True,
        ),
        parsing=lr.parsing.parser.ParsingConfig(
            separators=["\n\n"],
            splitter=lr.parsing.parser.Splitter.SIMPLE,
            n_similar_docs=2,
        )
    )
    agent = lr.agent.special.DocChatAgent(config)
    agent.ingest_docs(documents)
    task = lr.Task(
        agent=agent,
        default_human_response="summarize docs",
        interactive=False)
    task.run(turns=3)


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