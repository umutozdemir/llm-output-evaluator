# llm-output-evaluator
## Instructions
You can run the project locally and see your LLM's output quality. To do that, please do the following steps;
- `pip3 install requirements.txt`
- `python3 main.py`
  
Please make sure to add the documents you used to train your LLM into the dataset folder.

Please make sure that change the `OPENAI_API_KEY` value with your API key from OpenAI.

This project currently tests the custom LLM which I've created in `rag_pipeline.py`. So if you would like to test your LLM, you need to make modifications to the main.py/rag_pipeline.py

This project was developed for POC of the possibility of integrating output evaluators to the continuous deployment pipeline such as GitHub Actions.
