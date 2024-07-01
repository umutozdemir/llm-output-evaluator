from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator


# Checks if there is a hallucination
def test_faithfulness(response):
    evaluator = FaithfulnessEvaluator()
    eval_result = evaluator.evaluate_response(response=response)
    print(str(eval_result.passing))
    print(str(eval_result.score))
    print(str(eval_result.feedback))


def test_relevancy(query, response):
    evaluator = RelevancyEvaluator()
    eval_result = evaluator.evaluate_response(query=query, response=response)
    print(str(eval_result))
