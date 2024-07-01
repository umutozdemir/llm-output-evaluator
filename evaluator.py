from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import (AnswerRelevancyMetric, HallucinationMetric, FaithfulnessMetric, GEval,
                              ContextualPrecisionMetric, ContextualRecallMetric, ContextualRelevancyMetric)
from tabulate import tabulate
from textwrap import shorten

from const import OPEN_AI_MODEL


class Evaluator:
    def __init__(self, query, actual_output, expected_output, context, retrieval_context, threshold=0.5, model=OPEN_AI_MODEL):
        self.LLMTestCase = LLMTestCase(
            input=query,
            actual_output=actual_output,
            expected_output=expected_output,
            context=context,
            retrieval_context=retrieval_context,
        )
        self.answer_relevancy_metric = AnswerRelevancyMetric(threshold=threshold, model=model, include_reason=True,)
        self.hallucination_metric = HallucinationMetric(threshold=threshold, model=model, include_reason=True)
        self.correctness_metric = GEval(name="Correctness",
                                        criteria="Determine whether the actual output is factually correct based on "
                                                 "the expected output.",
                                        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT,
                                                           LLMTestCaseParams.EXPECTED_OUTPUT,
                                                           LLMTestCaseParams.CONTEXT],)
        self.faithfulness_metric = FaithfulnessMetric(threshold=threshold, model=model, include_reason=True)
        self.contextual_precision_metric = ContextualPrecisionMetric(threshold=threshold, model=model, include_reason=True)
        self.contextual_recall_metric = ContextualRecallMetric(threshold=threshold, model=model, include_reason=True)
        self.contextual_relevancy_metric = ContextualRelevancyMetric(threshold=threshold, model=model, include_reason=True)

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

        input_summary = shorten(self.LLMTestCase.input, width=30, placeholder="...")

        # Prepare the scores for printing
        headers = ["Input", "Answer Relevancy", "Hallucination", "Correctness", "Faithfulness",
                   "Contextual Precision", "Contextual Recall", "Contextual Relevancy"]
        scores = [input_summary, answer_relevancy, hallucination, correctness, faithfulness,
                  contextual_precision, contextual_recall, contextual_relevancy]

        # Calculate column widths for alignment
        column_widths = [max(len(str(item)) for item in col) for col in zip(headers, scores)]
        column_widths = [min(width, 15) for width in column_widths]  # Limit max column width

        # Format header
        header_row = " | ".join(f"{headers[i]:<{column_widths[i]}}" for i in range(len(headers)))
        print(header_row)
        print("-" * len(header_row))

        # Format scores
        score_row = " | ".join(f"{str(scores[i]):<{column_widths[i]}}" for i in range(len(scores)))
        print(score_row)
