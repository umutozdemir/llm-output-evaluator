import json
import os


def parse_synthetic_data_to_dict(synthetic_data_folder):
    # Find the JSON file in the synthetic_data folder
    json_file = None
    for file_name in os.listdir(synthetic_data_folder):
        if file_name.endswith('.json'):
            json_file = os.path.join(synthetic_data_folder, file_name)
            break

    # If no JSON file is found, raise an error
    if json_file is None:
        raise FileNotFoundError("No JSON file found in the synthetic_data folder.")

    # Read and parse the JSON file
    with open(json_file, 'r') as file:
        data = json.load(file)

    parsed_dict = {}
    for item in data:
        input_text = item['input']
        parsed_dict[input_text] = {
            'actual_output': item['actual_output'],
            'expected_output': item['expected_output'],
            'context': item['context'],
            'source_file': item['source_file']
        }

    return parsed_dict
