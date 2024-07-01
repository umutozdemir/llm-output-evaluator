from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric

from const import OPEN_AI_MODEL


class Evaluator:
    def __init__(self, threshold=0.5, model=OPEN_AI_MODEL):
        self.answer_relevancy_metric = AnswerRelevancyMetric(threshold=threshold, model=model)

    def evaluate_answer_relevancy(self, input_text, actual_output):
        test_case = LLMTestCase(
            input=input_text,
            actual_output=actual_output
        )
        return self.answer_relevancy_metric
