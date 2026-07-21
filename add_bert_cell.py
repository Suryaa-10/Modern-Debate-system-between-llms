import json
import os

with open('multi_model_demo.ipynb', 'r') as f:
    data = json.load(f)

# Markdown cell
md_cell = {
 'cell_type': 'markdown',
 'metadata': {},
 'source': [
  '## BERTScore Consensus Evaluation\n',
  "This cell evaluates the semantic similarity of the models' outputs using **BERTScore**. It compares the 3 low-parameter OpenRouter models against the high-parameter Groq model. If they all semantically agree (F1 Score > 0.85), it suppresses the reasoning and prints ONLY the final agreed answer."
 ]
}

# Code cell
code_cell = {
 'cell_type': 'code',
 'execution_count': None,
 'metadata': {},
 'outputs': [],
 'source': [
  '!pip install bert-score\n',
  '\n',
  'import torch\n',
  'from bert_score import score\n',
  '\n',
  'async def evaluate_consensus():\n',
  '    prompt = "What is the largest planet in our solar system and what is it mostly made of?"\n',
  '    print(f"Evaluating Consensus for Prompt: \'{prompt}\'\\n")\n',
  '    \n',
  '    low_param_models = [\n',
  '        "meta-llama/llama-4-maverick",\n',
  '        "mistralai/mistral-small-2603",\n',
  '        "google/gemma-4-26b-a4b-it:free"\n',
  '    ]\n',
  '    high_param_model = "llama-3.3-70b-versatile"\n',
  '    \n',
  '    tasks = []\n',
  '    for model in low_param_models:\n',
  '        tasks.append(get_response(openrouter_client, model, prompt))\n',
  '    tasks.append(get_response(groq_client, high_param_model, prompt))\n',
  '    \n',
  '    print("Generating responses...")\n',
  '    results = await asyncio.gather(*tasks)\n',
  '    \n',
  '    groq_ans = results[3]\n',
  '    low_param_ans = results[:3]\n',
  '    \n',
  '    print("Computing BERTScore...")\n',
  '    # Compare each low-param model to the high-param model\n',
  '    refs = [groq_ans] * 3\n',
  '    P, R, F1 = score(low_param_ans, refs, lang="en", verbose=False)\n',
  '    \n',
  '    print("\\nBERTScore F1 for each low-param model vs Groq model:")\n',
  '    all_match = True\n',
  '    for model, f1_score in zip(low_param_models, F1):\n',
  '        score_val = f1_score.item()\n',
  '        print(f"{model}: {score_val:.4f}")\n',
  '        if score_val < 0.85:\n',
  '            all_match = False\n',
  '            \n',
  '    if all_match:\n',
  '        print("\\n✅ CONSENSUS REACHED! All models strongly agree.")\n',
  '        print(f"Final Answer:\\n{groq_ans}")\n',
  '    else:\n',
  '        print("\\n❌ CONSENSUS FAILED! The models disagree.")\n',
  '        print("Outputs:")\n',
  '        for m, r in zip(low_param_models + [high_param_model], results):\n',
  '            print(f"[{m}]\\n{r}\\n")\n',
  '\n',
  'await evaluate_consensus()'
 ]
}

data['cells'].extend([md_cell, code_cell])

with open('multi_model_demo.ipynb', 'w') as f:
    json.dump(data, f, indent=1)

print('Added BERTScore cells')
