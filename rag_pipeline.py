from llama_index.core import VectorStoreIndex, Settings, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI
from const import OPEN_AI_MODEL


def create_new_query_engine():
    documents = SimpleDirectoryReader("./dataset").load_data()

    Settings.llm = OpenAI(model=OPEN_AI_MODEL)

    # bge-base embedding model
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

    index = VectorStoreIndex.from_documents(
        documents,
    )

    query_engine = index.as_query_engine()
    return query_engine


def get_responses_from_llm(queries, query_engine):
    responses = {}
    for query in queries:
        response = query_engine.query(query)
        responses[query] = {"answer": response.response,
                            "retrieval_context": [node.get_content() for node in response.source_nodes]}
    return responses
