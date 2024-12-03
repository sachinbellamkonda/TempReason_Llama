import json
import re
import networkx as nx
import dateparser
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from collections import Counter
import torch
from peft import PeftModel

# ------------------------------
# Evaluation Metrics
# ------------------------------

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        return re.sub(r'[^\w\s]', '', text)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def get_tokens(s, tokenizer):
    if not s:
        return []
    return tokenizer.tokenize(normalize_answer(s))

def compute_exact(a_gold, a_pred, tokenizer):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))

def compute_f1(a_gold, a_pred, tokenizer):
    # Use tokenizer-based tokens
    gold_toks = get_tokens(a_gold, tokenizer)
    pred_toks = get_tokens(a_pred, tokenizer)
    common = Counter(gold_toks) & Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If both are empty, then F1 is 1. If one is empty, F1 is 0
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = num_same / len(pred_toks)
    recall = num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

# ------------------------------
# Load Dataset
# ------------------------------

def load_dataset(json_path):
    data = []
    with open(json_path, 'r', encoding='utf-8') as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if line:
                try:
                    entry = json.loads(line)
                    data.append(entry)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON on line {line_number}: {e}")
                    print(f"Line content: {line}")
    return data

# ------------------------------
# Initialize Model and Tokenizer
# ------------------------------

def initialize_model(base_model_name, finetuned_model_path):
    """
    Initialize the tokenizer and model.
    :param base_model_name: Hugging Face model name of the base model.
    :param finetuned_model_path: Path to the fine-tuned model.
    :return: tokenizer, model
    """
    tokenizer = AutoTokenizer.from_pretrained(finetuned_model_path, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token  # Set pad_token_id

    # Load the base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="auto",
        load_in_8bit=True,
        torch_dtype=torch.float16,
    )

    # Load the PEFT fine-tuned model
    model = PeftModel.from_pretrained(model, finetuned_model_path)
    model.to('cuda')
    model.eval()

    return tokenizer, model

# ------------------------------
# Event Extraction
# ------------------------------

def extract_events_with_dates(context, tokenizer, model, max_length=2048):
    prompt = (
        "Question: Extract all temporal events with their associated start and end dates from the following text, formatting each event as (Event, Start Date, End Date):\n\n"
        "Context: Sachin has studied in ASU from Jan,2023 to July,2024. He then joined Google in Aug, 2024 and left in Dec, 2025"
         "Events: (\"Sachin studies in ASU \", \"Jan 2023\", \"July 2024\"), (\"Sachin joined Google\", \"Aug 2024\", \"Dec 2025\")"

        "Question: Extract all temporal events with their associated start and end dates from the following text, "
        "formatting each event as (Event, Start Date, End Date):\n\n Context: " + context + "\n\nEvents:"
    )

    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device)
    input_length = inputs['input_ids'].shape[1]

    outputs = model.generate(
        **inputs,
        max_length=min(max_length, input_length + 512),
        num_return_sequences=1,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Remove the prompt from the response
    response = response[len(prompt):].strip()

    print("\n\nExtracted event response:\n", response)

    # Parse the response to extract events
    pattern = r'\(([^,]+),\s*([^,]+),\s*([^)]+)\)'
    events = re.findall(pattern, response)
    return events

# ------------------------------
# Date Normalization
# ------------------------------

def parse_date(date_str):
    if date_str.lower() in ['present', 'current']:
        return "Present"
    parsed_date = dateparser.parse(date_str)
    return parsed_date.strftime('%Y-%m-%d') if parsed_date else None

def normalize_events(events):
    normalized = []
    for event, start, end in events:
        start_normalized = parse_date(start)
        end_normalized = parse_date(end)
        if start_normalized and end_normalized:
            normalized.append((event.strip(), start_normalized, end_normalized))
    return normalized

# ------------------------------
# Event Verification
# ------------------------------

def verify_event_range(event, start_date, end_date, tokenizer, model):
    user_prompt = (
        f"Consider the event: (Event: {event}, Start Date: {start_date}, End Date: {end_date}).\n"
        "Is this a complete and accurate representation of the event? Respond with 'yes' or 'no'."
    )
    prompt = user_prompt

    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device)
    outputs = model.generate(
        **inputs,
        max_length=512,
        num_return_sequences=1,
        temperature=0.0,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Get the model's response after the prompt
    response = response[len(prompt):].strip().lower()
    # Simple logic to decide if the event is verified
    if "no" in response:
        # Here, you can implement more sophisticated correction mechanisms
        # For simplicity, we'll log the issue and proceed
        print(f"Verification failed for event: {event}")
        return None
    else:
        return (event, start_date, end_date)

def verify_and_normalize_events(events, tokenizer, model):
    verified = []
    for event, start, end in events:
        verified_event = verify_event_range(event, start, end, tokenizer, model)
        if verified_event:
            verified.append(verified_event)
    return verified

# ------------------------------
# Temporal Graph Construction
# ------------------------------

def construct_temporal_graph(events):
    G = nx.DiGraph()
    event_nodes = []

    parsed_events = []
    for idx, (event, start, end) in enumerate(events):
        start_dt = dateparser.parse(start)
        end_dt = dateparser.parse(end) if end.lower() != "present" else dateparser.parse("today")
        parsed_events.append((idx, event, start_dt, end_dt))

    # Sort events by start date
    parsed_events.sort(key=lambda x: x[2])

    for idx, event, start_dt, end_dt in parsed_events:
        G.add_node(idx, event=event, start=start_dt, end=end_dt)
        event_nodes.append((idx, event, start_dt, end_dt))

    # Define relationships
    for i in range(len(event_nodes)):
        for j in range(i+1, len(event_nodes)):
            idx_a, event_a, start_a, end_a = event_nodes[i]
            idx_b, event_b, start_b, end_b = event_nodes[j]

            if end_a < start_b:
                relation = 'before'
            elif start_a > end_b:
                relation = 'after'
            elif start_a <= start_b and end_a >= end_b:
                relation = 'during'
            elif start_a < end_b and end_a > start_b:
                relation = 'overlaps'
            elif start_a == start_b and end_a == end_b:
                relation = 'simultaneous'
            else:
                relation = 'unknown'

            if relation != 'unknown':
                G.add_edge(idx_a, idx_b, relation=relation)

    return G

# ------------------------------
# Answer Generation
# ------------------------------

def generate_answer(question, temporal_graph, tokenizer, model, num_answers=3):
    nodes_data = "\n".join([
        f"Node {n}: {d['event']} ({d['start'].date()} to {d['end'].date()})"
        for n, d in temporal_graph.nodes(data=True)
    ])
    edges_data = "\n".join([
        f"{u} --{d['relation']}--> {v}"
        for u, v, d in temporal_graph.edges(data=True)
    ])
    prompt = (
        f"Using the following temporal graph data:\n"
        f"{nodes_data}\n\n"
        f"{edges_data}\n\n"
        f"Answer the question: \"{question}\"\n\nAnswer:"
    )

    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device)
    input_length = inputs['input_ids'].shape[1]

    outputs = model.generate(
        **inputs,
        max_length=min(512, input_length + 128),
        num_return_sequences=num_answers,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )
    # Decode and collect answers
    responses = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    answers = [response[len(prompt):].strip() for response in responses]

    # Implement self-consistency by selecting the most common answer
    most_common = Counter(answers).most_common(1)
    if most_common:
        return most_common[0][0]
    else:
        return "I'm unable to determine the answer based on the provided information."

# ------------------------------
# Answer Metrics
# ------------------------------

def calculate_metrics(predictions, ground_truths, tokenizer):
    exact_matches = 0
    f1_scores = 0.0
    total = len(predictions)

    for pred, gt_list in zip(predictions, ground_truths):
        em_list = []
        f1_list = []
        for gt in gt_list:
            em = compute_exact(gt, pred, tokenizer)
            f1 = compute_f1(gt, pred, tokenizer)
            em_list.append(em)
            f1_list.append(f1)
        exact_matches += max(em_list)
        f1_scores += max(f1_list)

    exact_match = exact_matches / total * 100
    f1 = f1_scores / total * 100
    return exact_match, f1

# ------------------------------
# Processing Functions
# ------------------------------

def process_entry(entry, tokenizer, model):
    print("Entry keys:", list(entry.keys()))
    context = entry.get('context', '')
    question = entry.get('question', '')
    ground_truths = entry.get('text_answers', {}).get('text', [])

    if not context:
        print(f"Entry ID {entry.get('id', 'Unknown')} has no context.")
        return {
            'id': entry.get('id', 'Unknown'),
            'question': question,
            'generated_answer': "No context provided.",
            'ground_truths': ground_truths,
            'exact_match': 0,
            'f1_score': 0.0
        }

    # Step 1: Event Extraction
    extracted_events = extract_events_with_dates(context, tokenizer, model)
    normalized_events = normalize_events(extracted_events)

    if not normalized_events:
        print(f"No events extracted for entry ID {entry.get('id', 'Unknown')}.")
        return {
            'id': entry.get('id', 'Unknown'),
            'question': question,
            'generated_answer': "Unable to extract events.",
            'ground_truths': ground_truths,
            'exact_match': 0,
            'f1_score': 0.0
        }

    # Step 2: Event Verification
    verified_events = verify_and_normalize_events(normalized_events, tokenizer, model)
    if not verified_events:
        return {
            'id': entry.get('id', 'Unknown'),
            'question': question,
            'generated_answer': "Unable to extract reliable events.",
            'ground_truths': ground_truths,
            'exact_match': 0,
            'f1_score': 0.0
        }

    # Step 3: Temporal Graph Construction
    temporal_graph = construct_temporal_graph(verified_events)

    # Step 4: Answer Generation
    generated_answer = generate_answer(question, temporal_graph, tokenizer, model)

    # Step 5: Evaluation Metrics
    em, f1 = calculate_metrics([generated_answer], [ground_truths], tokenizer)

    return {
        'id': entry.get('id', 'Unknown'),
        'question': question,
        'generated_answer': generated_answer,
        'ground_truths': ground_truths,
        'exact_match': em,
        'f1_score': f1
    }

# ------------------------------
# Main Processing Pipeline
# ------------------------------

def main():
    # Paths (Update these paths as needed)
    input_json_path = 'test_l3.json'
    output_json_path = 'processed_test_l3_results.json'
    base_model_name = 'meta-llama/Llama-3.1-8B'  # Base model name
    finetuned_model_path = 'llama_finetuned_user_peft'  # Path to your fine-tuned model

    # Load Dataset
    print("Loading dataset...")
    data = load_dataset(input_json_path)
    print(f"Loaded {len(data)} entries.")

    # Initialize Model and Tokenizer
    print("Initializing model and tokenizer...")
    tokenizer, model = initialize_model(base_model_name, finetuned_model_path)

    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Model is using device: {device}")

    # Process Entries
    results = []
    predictions = []
    ground_truths_all = []

    print("Processing entries...")
    for entry in tqdm(data, desc="Processing Entries"):
        result = process_entry(entry, tokenizer, model)
        results.append(result)
        # For overall metrics
        predictions.append(result['generated_answer'])
        ground_truths_all.append(result['ground_truths'])

    # Calculate Overall Metrics
    print("Calculating overall metrics...")
    overall_em, overall_f1 = calculate_metrics(predictions, ground_truths_all, tokenizer)
    print(f"Overall Exact Match (EM): {overall_em:.2f}%")
    print(f"Overall F1 Score: {overall_f1:.2f}%")

    # Save Results
    print(f"Saving results to {output_json_path}...")
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False, default=str)
    print("Processing completed.")

if __name__ == "__main__":
    main()
