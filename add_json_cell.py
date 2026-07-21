import json
import os

with open('multi_model_demo.ipynb', 'r') as f:
    data = json.load(f)

# Markdown cell
md_cell = {
 'cell_type': 'markdown',
 'metadata': {},
 'source': [
  '## JSON Output Enforcement\n',
  'This cell demonstrates how to enforce structured JSON outputs using `response_format={"type": "json_object"}`. It ensures the model responds exactly with the keys `"reasoning"` and `"answer"`.'
 ]
}

# Code cell
code_cell = {
 'cell_type': 'code',
 'execution_count': None,
 'metadata': {},
 'outputs': [],
 'source': [
  'async def get_json_response(client, model, prompt):\n',
  '    system_prompt = """You are a logical assistant. You must respond with a valid JSON object containing exactly two keys:\n',
  '1. \\"reasoning\\": a step-by-step explanation of your thought process.\n',
  '2. \\"answer\\": the final concise answer.\n',
  'Output ONLY valid JSON."""\n',
  '    try:\n',
  '        response = await client.chat.completions.create(\n',
  '            model=model,\n',
  '            messages=[\n',
  '                {"role": "system", "content": system_prompt},\n',
  '                {"role": "user", "content": prompt}\n',
  '            ],\n',
  '            response_format={"type": "json_object"}\n',
  '        )\n',
  '        return response.choices[0].message.content\n',
  '    except Exception as e:\n',
  '        return f"Error: {e}"\n',
  '\n',
  'async def json_demo():\n',
  '    prompt = "What is the capital of France and why is it important?"\n',
  '    print(f"Prompt: \'{prompt}\'\\n")\n',
  '    \n',
  '    tasks = []\n',
  '    for model in low_param_models:\n',
  '        tasks.append(get_json_response(openrouter_client, model, prompt))\n',
  '    tasks.append(get_json_response(groq_client, high_param_model, prompt))\n',
  '    \n',
  '    results = await asyncio.gather(*tasks)\n',
  '    \n',
  '    print("=== OpenRouter (JSON Output) ===")\n',
  '    for model, res in zip(low_param_models, results[:3]):\n',
  '        print(f"\\n[ {model} ]")\n',
  '        try:\n',
  '            parsed = json.loads(res)\n',
  '            print(json.dumps(parsed, indent=2))\n',
  '        except:\n',
  '            print(res)\n',
  '            \n',
  '    print("\\n=== Groq (JSON Output) ===")\n',
  '    print(f"\\n[ {high_param_model} ]")\n',
  '    try:\n',
  '        parsed = json.loads(results[3])\n',
  '        print(json.dumps(parsed, indent=2))\n',
  '    except:\n',
  '        print(results[3])\n',
  '\n',
  'await json_demo()'
 ]
}

data['cells'].extend([md_cell, code_cell])

with open('multi_model_demo.ipynb', 'w') as f:
    json.dump(data, f, indent=1)

print('Added JSON cells')
