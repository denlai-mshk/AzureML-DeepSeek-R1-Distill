import json
import os
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

# Global variables for model and tokenizer
model = None
tokenizer = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def init():
    global model, tokenizer
    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR", "."), "deepseek-model-qwen-1o5b")
    print(f"[DEBUG] Loading model from path: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, ignore_mismatched_sizes=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, ignore_mismatched_sizes=True)
    
    model.to(device)
    model.eval()

def run(data):
    try:
        if isinstance(data, str):
            data = json.loads(data)
        prompt = data.get("prompt", "")
        temperature_user = data.get("temperature", 0.7)
        top_p_user = data.get("top_p", 0.9)
        do_sample_user = data.get("do_sample", True)

        max_length_constant = 16384

        # Tokenize with explicit settings from tokenizer_config.json
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length_constant
        ).to(device)

        input_length = inputs['input_ids'].shape[-1]
        max_new_tokens_goodfit = min(data.get("max_new_tokens", 100), max_length_constant - input_length)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens_goodfit,
                do_sample=do_sample_user,
                temperature=temperature_user,
                top_p=top_p_user
            )

        # Extract only the generated portion (excluding the prompt)
        generated_tokens = outputs[0][inputs['input_ids'].shape[-1]:]
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=False)

        return json.dumps({"generated_text": generated_text})

    except Exception as e:
        return json.dumps({"error": f"Inference failed: {str(e)}"})
