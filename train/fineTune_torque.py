# fFine tuning with original + torque dataset

import os
import json
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset, Dataset, concatenate_datasets
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


## Load the Tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B", use_fast=False)
tokenizer.pad_token = tokenizer.eos_token  # Set pad_token_id


## Prepare the Training Datasets

def prepare_user_data(filenames):
    texts = []
    for filename in filenames:
        with open(filename, 'r') as f:
            first_char = f.read(1)
            f.seek(0)
            if first_char == '[':
                # Standard JSON array
                data = json.load(f)
                for example in data:
                    question = example['question']
                    answer = example['text_answers']['text'][0]
                    text = f"Question: {question}\nAnswer: {answer}"
                    texts.append(text)
            else:
                # JSON Lines format
                for line in f:
                    if line.strip():
                        example = json.loads(line)
                        question = example['question']
                        answer = example['text_answers']['text'][0]
                        text = f"Question: {question}\nAnswer: {answer}"
                        texts.append(text)
    return {'text': texts}

def prepare_torque_data(filename):
    prompts = []
    responses = []
    with open(filename, 'r') as f:
        data = json.load(f)  # Load the JSON array
        for idx, item in enumerate(data):
            if 'passages' in item and item['passages']:
                for passage_item in item['passages']:
                    passage = passage_item.get('passage', '')
                    if 'question_answer_pairs' in passage_item and passage_item['question_answer_pairs']:
                        for qa_pair in passage_item['question_answer_pairs']:
                            question = qa_pair.get('question', '')
                            answer_spans = qa_pair.get('answer', {}).get('spans', [])
                            answer_text = ' '.join(answer_spans)
                            prompt = f"Passage: {passage}\nQuestion: {question}\nAnswer:"
                            response = answer_text
                            prompts.append(prompt)
                            responses.append(response)
    return {'text': [f"{p} {r}" for p, r in zip(prompts, responses)]}

# Prepare datasets
user_filenames = ['train_l1.json', 'train_l2.json', 'train_l3.json']
user_data_prepared = prepare_user_data(user_filenames)
user_dataset = Dataset.from_dict(user_data_prepared)

torque_filename = 'train_tq.json'  
torque_data_prepared = prepare_torque_data(torque_filename)
torque_dataset = Dataset.from_dict(torque_data_prepared)

# Concatenate datasets
combined_dataset = concatenate_datasets([user_dataset, torque_dataset])

# Check dataset sizes
print(f"Number of user training samples: {len(user_dataset)}")
print(f"Number of TORQUE training samples: {len(torque_dataset)}")
print(f"Total training samples: {len(combined_dataset)}")


## Tokenize the Dataset

def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, max_length=256)

tokenized_dataset = combined_dataset.map(tokenize_function, batched=True, remove_columns=['text'])


## Load the Model and Apply LoRA

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B",
    device_map="auto",
    load_in_8bit=True,
    torch_dtype=torch.float16,
)
model = prepare_model_for_kbit_training(model)
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.to("cuda")


##Set Up Training Arguments

# Adjusted Training Arguments
training_args = TrainingArguments(
    output_dir='./results_combined',
    max_steps=10000,
    per_device_train_batch_size=4,  
    gradient_accumulation_steps=4,
    learning_rate=1e-4,
    fp16=True,
    save_total_limit=2,
    logging_steps=50,
    save_steps=200,
    report_to='none',
    gradient_checkpointing=True,  
)



## Initialize the Trainer

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)


## Start Training

torch.cuda.empty_cache()
trainer.train()



## Save the Final Model

model.save_pretrained('llama_finetuned_combined_peft')
tokenizer.save_pretrained('llama_finetuned_combined_peft')
