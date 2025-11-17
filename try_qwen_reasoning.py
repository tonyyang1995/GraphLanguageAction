import pandas as pd
from tqdm import tqdm

import torch

import os
from transformers import AutoModelForCausalLM, AutoTokenizer

HF_TOKEN= os.environ["HF_API_KEY"]
df = pd.read_csv("datasets/chembl2k/raw/assays.csv.gz")
columns = df.columns.tolist()
gene = columns[4:]

print(len(gene))

smiles = df["smiles"].tolist()
print(len(smiles))

def generate_reasoning_smile_gene_local(smile, genes, model, tokenizer):
    genes_prompt = ','.join(genes)
    prompt = (
        f"Given the following molecule represented by the SMILES string: {smile}, "
        f"and the following genes: {genes_prompt}, "
        "describe whether the molecule is related to one or a few of these genes in clear, simple and accurate English. "
        "Explain the relationship between the molecule and the related genes in detail. "
        "You can break down by explain the patterns of the molecule, followed by the gene features. "
        "Format the output as 'Step 1:', 'Step 2:', etc., with each step focusing on a specific aspect of the molecule's structure or properties. "
        "Finally, summarize the related gene names in a list. "
        "Limit the reasoning to at most 5 steps. You can stop after summarizing the related gene names."
    )
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    attention_mask = (input_ids != tokenizer.pad_token_id).long() if tokenizer.pad_token_id is not None else None

    # use cuda
    input_ids = input_ids.to("cuda")
    attention_mask = attention_mask.to("cuda")
    model = model.to("cuda")
    
    model.eval()
    
    output_ids = model.generate(input_ids, attention_mask=attention_mask, max_length=1024)
    # Post-processing to avoid repeated contents in the output
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Simple heuristic: remove exact duplicate consecutive lines and repeated sections using set
    # This is a basic approach; for more advanced deduplication, consider NLP-based segmentation.
    def deduplicate_lines(text):
        lines = text.split('\n')
        seen = set()
        deduped = []
        for line in lines:
            if line.strip() and line not in seen:
                deduped.append(line)
                seen.add(line)
        return '\n'.join(deduped)

    output_text = deduplicate_lines(output_text)
    output = output_text
    
    # output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return output

def generate_reasoning_gene_local(gene, model, tokenizer):
    prompt = (
        f"Given the following gene: {gene}, "
        "describe its function in clear, simple and accurate English. "
        "Analyze the gene step by step, starting from an overview or definition, "
        "and then breaking down its key features in detail. "
        "Format the output as 'Step 1:', 'Step 2:', etc., with each step focusing on a specific aspect of the gene's function or properties. "
        "Do not repeat the question. Answer only in English functional descriptions."
    )
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    attention_mask = (input_ids != tokenizer.pad_token_id).long() if tokenizer.pad_token_id is not None else None

    # use cuda
    input_ids = input_ids.to("cuda")
    attention_mask = attention_mask.to("cuda")
    model = model.to("cuda")

    model.eval()

    output_ids = model.generate(input_ids, attention_mask=attention_mask, max_length=1024)

    output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return output

def generate_reasoning_smile_local(smile, model, tokenizer):
    prompt = (
        f"Given the following molecule represented by the SMILES string: {smile}, "
        "describe its structure in clear, simple and accurate English. "
        "Analyze the molecule step by step, starting from an overview or definition, "
        "and then breaking down its key structural features in detail. "
        "Format the output as 'Step 1:', 'Step 2: ..., 'Step 5: ...', 'Summary: ...'."
        "Focus on the structure of the molecule and the effect of related genes."
        "Do not repeat the question. Answer only in English structural descriptions."
    )
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    attention_mask = (input_ids != tokenizer.pad_token_id).long() if tokenizer.pad_token_id is not None else None

    # use cuda
    input_ids = input_ids.to("cuda")
    attention_mask = attention_mask.to("cuda")
    model = model.to("cuda")
    
    model.eval()

    output_ids = model.generate(input_ids, attention_mask=attention_mask, max_length=1024)

    output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return output

model_name = "Qwen/Qwen2.5-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(model_name, token=HF_TOKEN)
os.makedirs(f"reasoning_results/smiles", exist_ok=True)

for smile in tqdm(smiles):
    if os.path.exists(f"reasoning_results/smiles/{smile}.txt"):
        continue
    output = generate_reasoning_smile_gene_local(smile, gene, model, tokenizer)

    with open(f"reasoning_results/smiles/{smile}.txt", "w") as f:
        f.write(output)
        f.write("\n")
        f.close()
