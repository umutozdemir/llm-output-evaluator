from golden_standard_generator import generate_golden_standard
from utils import parse_synthetic_data_to_dict
from rag_pipeline import create_new_query_engine, get_responses_from_llm
from evaluator import Evaluator

if __name__ == '__main__':
    # First step: Generate golden standard.
    generate_golden_standard()

    # Second step: Read golden standard.
    queries = parse_synthetic_data_to_dict("./synthetic_data")

    # Third step: Create a query engine from RAG pipeline.
    query_engine = create_new_query_engine()

    # Forth step: Ask queries to LLM and create query->{answer, retrieval_context} dict.
    responses = get_responses_from_llm(queries, query_engine)

    # Fifth step: Evaluate the responses.
    for query in queries:
        response = responses.get(query)
        evaluator = Evaluator(query, response.get('answer'), queries.get(query).get("expected_output"),
                              queries.get(query).get("context"), response.get('retrieval_context'))

        evaluator.evaluate_output()
