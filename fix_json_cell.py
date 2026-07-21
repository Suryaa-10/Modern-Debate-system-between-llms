import json
import os

with open('multi_model_demo.ipynb', 'r') as f:
    data = json.load(f)

for cell in data['cells']:
    if cell['cell_type'] == 'code' and 'async def json_demo():' in ''.join(cell['source']):
        source = cell['source']
        for i, line in enumerate(source):
            if 'async def json_demo():\n' == line:
                # Insert the model definitions right after the function definition
                insert_code = [
                    '    low_param_models = [\n',
                    '        "meta-llama/llama-4-maverick",\n',
                    '        "mistralai/mistral-small-2603",\n',
                    '        "google/gemma-4-26b-a4b-it:free"\n',
                    '    ]\n',
                    '    high_param_model = "llama-3.3-70b-versatile"\n'
                ]
                source = source[:i+1] + insert_code + source[i+1:]
                cell['source'] = source
                break

with open('multi_model_demo.ipynb', 'w') as f:
    json.dump(data, f, indent=1)

print('Fixed JSON cells')
