## Langroid setup.

<!-- Set up python venv -->
cd cs839-langroid
python3 -m venv .venv
. ./.venv/bin/activate

# Update .env file.
For summarization task, copy .env-template as .env and just update openAI key. Then comment rest keys.
OPENAI_API_KEY=<>

<!-- Install the module in an editable way -->
# Run below command from cs839-langroid/
pip3 install -e .

<!-- Now, hopefully, all invokations of langroid refer to cs839-langroid/langroid -->

<!-- Run default langroid summarization -->
python3 examples/summarization/langroid_test.py

<!-- Run default summarization_agent baseline -->
# Summarization's summarize_chunked_docs implements the trivial baseline for long text summarization.
# By default, the agent summarizes `Attention is all you need` paper. The output is also listed in the same repo.
python3 examples/summarization/summarization_baseline.py