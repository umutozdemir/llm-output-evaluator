from llama_index.core import VectorStoreIndex, Settings, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama


def create_new_query_engine():
    documents = SimpleDirectoryReader("./dataset").load_data()

    # bge-base embedding model
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

    # ollama
    Settings.llm = Ollama(model="llama3", request_timeout=360.0)

    # bge-base embedding model
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

    index = VectorStoreIndex.from_documents(
        documents,
    )

    query_engine = index.as_query_engine()
    return query_engine


def get_query_responses(queries, query_engine):
    responses = {}
    for query in queries:
        response = query_engine.query(query)
        responses[query] = response
    return responses
