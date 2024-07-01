from llama_index.core import VectorStoreIndex, Settings, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

import tester

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
response = query_engine.query("Can I go to store on 7 pm on sunday?")

tester.test_faithfulness(response)
tester.test_relevancy(response)
