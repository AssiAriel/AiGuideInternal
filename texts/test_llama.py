from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template # <--- ADD THIS LINE
import torch

# Direct path to your D: drive folder
model_path = "/mnt/d/ModelManagers/HuggingFace/Models"

print("🧠 Loading model into RTX 5070 Ti VRAM...")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_path, # Points to your local files
    max_seq_length = 2048,
    load_in_4bit = True,
)

# Manually set the Llama 3.1 template
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "llama-3.1",
)
# -----------------------------------------------

FastLanguageModel.for_inference(model)

# Ask a simple question
messages = [
    {"role": "user", "content": "Tell me a 1-sentence fun fact about space."}
]

inputs = tokenizer.apply_chat_template(
    messages,
    tokenize = True,
    add_generation_prompt = True,
    return_tensors = "pt",
).to("cuda")

print("\n✨ Generating response...\n")

outputs = model.generate(input_ids = inputs, max_new_tokens = 64)
response = tokenizer.batch_decode(outputs)

# Clean up the output to show just the answer
print(response[0].split("assistant")[-1].replace("<|end_of_text|>", "").strip())