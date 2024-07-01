from golden_standard_generator import generate_golden_standard
from utils import parse_synthetic_data_to_dict
from rag_pipeline import create_new_query_engine, get_query_responses

if __name__ == '__main__':
    # First step: Generate golden standard.
    generate_golden_standard()

    # Second step: Read golden standard.
    queries = parse_synthetic_data_to_dict("./synthetic_data")

    # Third step: Create a query engine from RAG pipeline.
    query_engine = create_new_query_engine()

    # Forth step: Ask queries to LLM and create query->answer dict.
    answers = get_query_responses(queries, query_engine)

    # Fifth step: Evaluate the responses.
