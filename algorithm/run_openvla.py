"""Run OpenVLA vanilla inference on a sample image."""

import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image

MODEL_ID = "openvla/openvla-7b"

print(f"Loading model: {MODEL_ID}")
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForVision2Seq.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation="eager",
)
model.eval()
print(f"Model loaded. Device: {model.device}")

# Use a simple test image (solid color placeholder)
image = Image.new("RGB", (224, 224), color=(128, 128, 128))

prompt = "In: What action should the robot take to pick up the object?\nOut:"

inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)

with torch.no_grad():
    output_ids = model.generate(**inputs, max_new_tokens=7, do_sample=False)

input_len = inputs["input_ids"].shape[1]
generated_ids = output_ids[:, input_len:]

output_text = processor.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
print(f"Prompt: {prompt}")
print(f"Generated action tokens: {generated_ids[0].tolist()}")
print(f"Decoded output: {output_text}")
