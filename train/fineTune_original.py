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


#Load the Tokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B", use_fast=False)
tokenizer.pad_token = tokenizer.eos_token  


#Prepare Your Training Datasets

def prepare_user_data(filenames):
    texts = []
    for filename in filenames:
        with open(filename, 'r') as f:
            for line in f:
                if line.strip():  
                    example = json.loads(line)
                    question = example['question']
                    answer = example['text_answers']['text'][0]
                    text = f"Question: {question}\nAnswer: {answer}"
                    texts.append(text)
    return {'text': texts}

user_filenames = ['train_l1.json', 'train_l2.json', 'train_l3.json']
user_data_prepared = prepare_user_data(user_filenames)
user_dataset = Dataset.from_dict(user_data_prepared)


#Tokenize the Dataset

def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, max_length=512)

tokenized_user_dataset = user_dataset.map(tokenize_function, batched=True, remove_columns=['text'])


# Load the Model and Apply LoRA

from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B",
    device_map="auto",
    load_in_8bit=True,
    torch_dtype=torch.float16,
)

model = prepare_model_for_kbit_training(model)

# Configure LoRA
lora_config = LoraConfig(
    r=16,  
    lora_alpha=32,  
    target_modules=["q_proj", "v_proj"],  
    lora_dropout=0.05,  
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)


#Set Up Training Arguments

training_args = TrainingArguments(
    output_dir='./results_user',
    max_steps=1000,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=1,  
    learning_rate=1e-4,
    fp16=True,
    save_total_limit=2,
    logging_steps=50,
    save_steps=200,
    report_to='none',
    gradient_checkpointing=False,
)





#Initialize the Trainer

from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_user_dataset,
    data_collator=data_collator,
)



trainer.train()

model.save_pretrained('llama_finetuned_user_peft')

tokenizer.save_pretrained('llama_finetuned_user_peft')