from evaluator import Evaluator
from utils import parse_synthetic_data_to_dict
from rag_pipeline import create_new_query_engine, get_responses_from_llm


def generate_llm_outputs():
    # Read golden standard.
    queries = parse_synthetic_data_to_dict("./synthetic_data")

    # Create a query engine from RAG pipeline.
    query_engine = create_new_query_engine()

    # Ask queries to LLM and create query->{answer, retrieval_context} dict.
    responses = get_responses_from_llm(queries, query_engine)
    return evaluate_llm_output(queries, responses)


def evaluate_llm_output(queries, responses):
    for query in queries:
        response = responses.get(query)
        evaluator = Evaluator(query, response.get('answer'), queries.get(query).get("expected_output"),
                              queries.get(query).get("context"), response.get('retrieval_context'))

        evaluator.evaluate_output()


generate_llm_outputs()
