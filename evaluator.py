import json

from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import (AnswerRelevancyMetric, HallucinationMetric, FaithfulnessMetric, GEval,
                              ContextualPrecisionMetric, ContextualRecallMetric, ContextualRelevancyMetric)

from const import OPEN_AI_MODEL


class Evaluator:
    def __init__(self, query, actual_output, expected_output, context, retrieval_context, threshold=0.5,
                 model=OPEN_AI_MODEL):
        self.LLMTestCase = LLMTestCase(
            input=query,
            actual_output=actual_output,
            expected_output=expected_output,
            context=context,
            retrieval_context=retrieval_context,
        )
        self.answer_relevancy_metric = AnswerRelevancyMetric(threshold=threshold, model=model, include_reason=True, )
        self.hallucination_metric = HallucinationMetric(threshold=threshold, model=model, include_reason=True)
        self.correctness_metric = GEval(name="Correctness",
                                        criteria="Determine whether the actual output is factually correct based on "
                                                 "the expected output.",
                                        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT,
                                                           LLMTestCaseParams.EXPECTED_OUTPUT,
                                                           LLMTestCaseParams.CONTEXT], )
        self.faithfulness_metric = FaithfulnessMetric(threshold=threshold, model=model, include_reason=True)
        self.contextual_precision_metric = ContextualPrecisionMetric(threshold=threshold, model=model,
                                                                     include_reason=True)
        self.contextual_recall_metric = ContextualRecallMetric(threshold=threshold, model=model, include_reason=True)
        self.contextual_relevancy_metric = ContextualRelevancyMetric(threshold=threshold, model=model,
                                                                     include_reason=True)
        self.results = []

    def evaluate_answer_relevancy(self):
        self.answer_relevancy_metric.measure(self.LLMTestCase)
        return [self.answer_relevancy_metric.score, self.answer_relevancy_metric.reason]

    def evaluate_hallucination(self):
        self.hallucination_metric.measure(self.LLMTestCase)
        return [self.hallucination_metric.score, self.hallucination_metric.reason]

    def evaluate_correctness(self):
        self.correctness_metric.measure(self.LLMTestCase)
        return [self.correctness_metric.score, self.correctness_metric.reason]

    def evaluate_faithfulness(self):
        self.faithfulness_metric.measure(self.LLMTestCase)
        return [self.faithfulness_metric.score, self.faithfulness_metric.reason]

    def evaluate_contextual_precision_metric(self):
        self.contextual_precision_metric.measure(self.LLMTestCase)
        return [self.contextual_precision_metric.score, self.contextual_precision_metric.reason]

    def evaluate_contextual_recall_metric(self):
        self.contextual_recall_metric.measure(self.LLMTestCase)
        return [self.contextual_recall_metric.score, self.contextual_recall_metric.reason]

    def evaluate_contextual_relevancy_metric(self):
        self.contextual_relevancy_metric.measure(self.LLMTestCase)
        return [self.contextual_relevancy_metric.score, self.contextual_relevancy_metric.reason]

    def evaluate_output(self):
        answer_relevancy = self.evaluate_answer_relevancy()
        hallucination = self.evaluate_hallucination()
        correctness = self.evaluate_correctness()
        faithfulness = self.evaluate_faithfulness()
        contextual_precision = self.evaluate_contextual_precision_metric()
        contextual_recall = self.evaluate_contextual_recall_metric()
        contextual_relevancy = self.evaluate_contextual_relevancy_metric()

        metrics = [
            ("Answer Relevancy", answer_relevancy),
            ("Hallucination", hallucination),
            ("Correctness", correctness),
            ("Faithfulness", faithfulness),
            ("Contextual Precision", contextual_precision),
            ("Contextual Recall", contextual_recall),
            ("Contextual Relevancy", contextual_relevancy)
        ]

        scores = []
        for field_name, (field_score, reason) in metrics:
            self.results.append({
                "input": self.LLMTestCase.input,
                "field_name": field_name,
                "field_score": field_score,
                "reason": reason
            })
            scores.append(field_score)
        print(self.results)
        results_json = json.dumps(self.results, indent=4)
        self.save_results('evaluation_results.json', results_json)

        return scores

    def print_results(self):
        print(f"{'Input':<50} {'Field Name':<20} {'Score':<10} {'Reason'}")
        print("=" * 100)
        for result in self.results:
            print(f"{result['input']:<50} {result['field_name']:<20} {result['field_score']:<10} {result['reason']}")

    @staticmethod
    def save_results(file_path, results):
        with open(file_path, 'w') as f:
            f.write(results)
