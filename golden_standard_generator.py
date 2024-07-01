import os
from deepeval.synthesizer import Synthesizer

os.environ["OPENAI_API_KEY"] = "sk-proj-KPBo2qv14PNo64Vo31XBT3BlbkFJNPUOW9djeUNxsBjoH8z4"


def generate_golden_dataset_deep_eval():
    # Get all file paths in the ./dataset directory
    document_paths = [os.path.join("./dataset", filename) for filename in os.listdir("./dataset") if
                      os.path.isfile(os.path.join("./dataset", filename))]

    synthesizer = Synthesizer(model="gpt-3.5-turbo")
    synthesizer.generate_goldens_from_docs(
        document_paths=document_paths,
        max_goldens_per_document=5,
        include_expected_output=True
    )

    synthesizer.save_as(
        file_type='json',
        directory="./synthetic_data",
    )


generate_golden_dataset_deep_eval()
