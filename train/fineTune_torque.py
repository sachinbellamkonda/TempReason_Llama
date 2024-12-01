# Fine tuning with original dataset.

import os
import json
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset, Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ------------------------------
# 1. Load the Tokenizer
# ------------------------------
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B", use_fast=False)
tokenizer.pad_token = tokenizer.eos_token  # Set pad_token_id

# ------------------------------
# 2. Prepare Your Training Datasets
# ------------------------------
def prepare_user_data(filenames):
    texts = []
    for filename in filenames:
        with open(filename, 'r') as f:
            for line in f:
                if line.strip():  # Skip empty lines
                    example = json.loads(line)
                    question = example['question']
                    answer = example['text_answers']['text'][0]
                    # Combine question and answer into a single text
                    text = f"Question: {question}\nAnswer: {answer}"
                    texts.append(text)
    return {'text': texts}

user_filenames = ['train_l1.json', 'train_l2.json', 'train_l3.json']
user_data_prepared = prepare_user_data(user_filenames)
user_dataset = Dataset.from_dict(user_data_prepared)

# ------------------------------
# 3. Tokenize the Dataset
# ------------------------------
def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, max_length=512)

tokenized_user_dataset = user_dataset.map(tokenize_function, batched=True, remove_columns=['text'])

# ------------------------------
# 4. Load the Model and Apply LoRA
# ------------------------------
from transformers import AutoModelForCausalLM

# Load the model in 8-bit mode
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B",
    device_map="auto",
    load_in_8bit=True,
    torch_dtype=torch.float16,
)

# Prepare the model for k-bit (8-bit) training
model = prepare_model_for_kbit_training(model)

# Configure LoRA
lora_config = LoraConfig(
    r=16,  # Rank of the update matrices
    lora_alpha=32,  # Scaling factor
    target_modules=["q_proj", "v_proj"],  # Modules to apply LoRA to
    lora_dropout=0.05,  # Dropout rate
    bias="none",
    task_type="CAUSAL_LM",
)

# Apply LoRA to the model
model = get_peft_model(model, lora_config)

# ------------------------------
# 5. Set Up Training Arguments
# ------------------------------
training_args = TrainingArguments(
    output_dir='./results_user',
    max_steps=1000,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=1,  # Adjust as needed
    learning_rate=1e-4,
    fp16=True,
    save_total_limit=2,
    logging_steps=50,
    save_steps=200,
    report_to='none',
    gradient_checkpointing=False,
)




# ------------------------------
# 6. Initialize the Trainer
# ------------------------------
from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_user_dataset,
    data_collator=data_collator,
)

# ------------------------------
# 7. Start Training
# ------------------------------
trainer.train()

# ------------------------------
# 8. Save the Final Model
# ------------------------------
# Save the PEFT model
model.save_pretrained('llama_finetuned_user_peft')

# Save the tokenizer
tokenizer.save_pretrained('llama_finetuned_user_peft')

# ------------------------------
# 9. Testing the Model
# ------------------------------
from peft import PeftModel

# Load the base model in 8-bit mode
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B",
    device_map="auto",
    load_in_8bit=True,
    torch_dtype=torch.float16,
)

# Load the PEFT model
model = PeftModel.from_pretrained(model, 'llama_finetuned_user_peft')

# Function to generate responses
def generate_responses(prompts):
    responses = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors='pt').to('cuda')
        output = model.generate(
            **inputs,
            max_new_tokens=100,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=False  # For deterministic output
        )
        # Extract the generated text after the prompt
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        # Remove the prompt from the generated text
        response = generated_text[len(prompt):].strip()
        responses.append(response)
    return responses

# Example usage:
test_prompts = [
    "Question: What is the time 6 year and 4 month after Nov, 1185?\nAnswer:",
    "Question: Which employer did Jaroslav Pelikan work for in Jan, 1948?\nAnswer:",
    # Add more prompts as needed
]

# Generate responses
responses = generate_responses(test_prompts)

# Print responses
for prompt, response in zip(test_prompts, responses):
    print(f"Prompt:\n{prompt}\nResponse:\n{response}\n{'-'*50}")
