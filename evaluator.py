from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric


class Evaluator:
    def evaluate_answer_relevancy(self, input, actual_output):
        answer_relevancy_metric = AnswerRelevancyMetric(threshold=0.5, model='gpt-3.5-turbo')
        test_case = LLMTestCase(
            input=input,
            # Replace this with the actual output of your LLM application
            actual_output=actual_output
        )
        return answer_relevancy_metric
