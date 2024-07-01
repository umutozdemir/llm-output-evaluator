import os
from deepeval.synthesizer import Synthesizer

from const import OPEN_AI_MODEL, OPEN_AI_API_KEY

os.environ["OPENAI_API_KEY"] = OPEN_AI_API_KEY


def generate_golden_standard():
    # Get all file paths in the ./dataset directory
    document_paths = [os.path.join("./dataset", filename) for filename in os.listdir("./dataset") if
                      os.path.isfile(os.path.join("./dataset", filename))]

    synthesizer = Synthesizer(model=OPEN_AI_MODEL)
    synthesizer.generate_goldens_from_docs(
        document_paths=document_paths,
        max_goldens_per_document=5,
        include_expected_output=True
    )

    synthesizer.save_as(
        file_type='json',
        directory="./synthetic_data",
    )


generate_golden_standard()
