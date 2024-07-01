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
    sum_of_metric_scores = [
        ("Answer Relevancy", 0),
        ("Hallucination", 0),
        ("Correctness", 0),
        ("Faithfulness", 0),
        ("Contextual Precision", 0),
        ("Contextual Recall", 0),
        ("Contextual Relevancy", 0)
    ]

    for query in queries:
        response = responses.get(query)
        evaluator = Evaluator(query, response.get('answer'), queries.get(query).get("expected_output"),
                              queries.get(query).get("context"), response.get('retrieval_context'))

        scores = evaluator.evaluate_output()

        for i, (field_name, score) in enumerate(sum_of_metric_scores):
            sum_of_metric_scores[i] = (field_name, score + scores[i])

    average_scores = [(field_name, score / len(queries)) for field_name, score in sum_of_metric_scores]

    # Print the average scores
    print("Average Scores:")
    for field_name, average_score in average_scores:
        print(f"{field_name}: {average_score:.2f}")


generate_llm_outputs()
