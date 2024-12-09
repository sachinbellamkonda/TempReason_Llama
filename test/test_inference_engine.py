import json
import os
import torch
import re
import networkx as nx
import dateparser
from transformers import AutoTokenizer
from tqdm import tqdm
from collections import Counter
import torch
from vllm import LLM, SamplingParams
import matplotlib.pyplot as plt  

# Evaluation Metrics

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
    gold_toks = get_tokens(a_gold, tokenizer)
    pred_toks = get_tokens(a_pred, tokenizer)
    common = Counter(gold_toks) & Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = num_same / len(pred_toks)
    recall = num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

# Load Dataset

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

# Date Normalization

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

def parse_events(cot_response):
    tuple_pattern = r'\(\s*"(.*?)"\s*,\s*"(.*?)"\s*,\s*"(.*?)"\s*\)'
    cleaned_response = re.sub(r'\s*\n\s*', ' ', cot_response)  # Handle multiline cases
    matches = re.findall(tuple_pattern, cleaned_response)
    events = []
    for match in matches:
        event = tuple(field.strip() for field in match)
        events.append(event)
    return events

# Temporal Graph Construction

def construct_temporal_graph(events):
    G = nx.DiGraph()
    event_nodes = []
    parsed_events = []
    for idx, (event, start, end) in enumerate(events):
        start_dt = dateparser.parse(start) if start != "Present" else dateparser.parse("today")
        end_dt = dateparser.parse(end) if end != "Present" else dateparser.parse("today")
        parsed_events.append((idx, event, start_dt, end_dt))

    for idx, event, start_dt, end_dt in parsed_events:
        G.add_node(idx, event=event, start=start_dt, end=end_dt)
        event_nodes.append((idx, event, start_dt, end_dt))

    for i in range(len(event_nodes)):
        for j in range(i+1, len(event_nodes)):
            idx_a, event_a, start_a, end_a = event_nodes[i]
            idx_b, event_b, start_b, end_b = event_nodes[j]

            if start_a <= start_b and end_a >= end_b:
                relation = 'during'
            elif start_a < end_b and end_a > start_b:
                relation = 'overlaps'
            elif start_a == start_b and end_a == end_b:
                relation = 'simultaneous'
            elif end_a < start_b:
                relation = 'before'
            elif start_a > end_b:
                relation = 'after'
            else:
                relation = 'unknown'

            if relation != 'unknown':
                G.add_edge(idx_a, idx_b, relation=relation)
    return G

# Graph Visualization Function

def visualize_graph(G, filename):
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, k=0.5, iterations=50)
    node_labels = {node: data['event'] for node, data in G.nodes(data=True)}
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color='lightblue')
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8)
    edge_labels = nx.get_edge_attributes(G, 'relation')
    nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=20, edge_color='gray')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', font_size=8)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# Answer Metrics

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

# Processing Functions

def process_entries(entries, llm, tokenizer):
    batch_size = len(entries)
    results = []
    event_extraction_prompts = []
    entry_ids = []
    questions = []
    ground_truths_list = []
    contexts = []

    for entry in entries:
        context = entry.get('context', '')
        fact_context = entry.get('fact_context','')
        question = entry.get('question', '')
        ground_truths = entry.get('text_answers', {}).get('text', [])
        entry_id = entry.get('id', 'Unknown')

        if not context:
            results.append({
                'id': entry_id,
                'question': question,
                'generated_answer': "No context provided.",
                'ground_truths': ground_truths,
                'exact_match': 0,
                'f1_score': 0.0
            })
            continue
# event extrction prompt
        prompt = (
            """Question: extract temporal events in the (Event, Start, End) format from the text above, mention the subject involved at the start of each event sentence, return as a list of tuples\n\n"
            "Context: Norman BrunsdaleClarence Norman Brunsdale (July 9, 1891January 27, 1978) was the 24th Governor of North Dakota and a United States Senator from the state of North Dakota.Clarence Norman Brunsdale was born in Sherbrooke, Steele County, North Dakota. he was the son of Knute H. Brunsdale (1855-1899) and Anna Margaret (Nordgaard) Brunsdale (1860-1927), both of whom were of Norwegian immigrant heritage. He was educated in public schools and the Bruflat Academy at Portland, North Dakota. In 1913, he graduated from Luther College in Decorah, Iowa. He returned to Portland, teaching at Bruflat Academy and worked the family farm operations in Traill and Steele counties.Brunsdale served in the North Dakota State Senate (1927–34, 1941–51). He was an alternate delegate to Republican National Convention from North Dakota (1940) and a member of Republican National Committee from North Dakota, (1948–52). He was Governor of North Dakota from 1951 to 1957 and U.S. Senator from November 19, 1959 to August 7, 1960. As governor, Brunsdale was an avid supporter of water development projects. During his administration Garrison Dam was completed and the Legislature established the Garrison Diversion Conservancy District. The early 1950s also saw the establishment of the Highway Department and the passage of major highway legislation. Education, agriculture, and mental health issues were also important to Governor Brunsdale. In 1959, Brunsdale was appointed to the United States Senate upon the death of Senator William Langer. Brunsdale voted in favor of the Civil Rights Act of 1960. Brunsdale was not a candidate for election to the vacancy and Quentin Burdick was narrowly elected to the seat in a 1960 special election.He was married to Carrie Lajord (1890-1982) on August 30, 1925, and they had two daughters, Margaret Marie (Larson) and Helen Lucille (Williams). Brunsdale died at Mayville, North Dakota in 1978. He was buried in Mayville Cemetery, Mayville, Traill County, North Dakota."
            "Events: [
    ("Clarence Norman Brunsdale's birth", "July 9, 1891", "July 9, 1891"),
    ("Knute H. Brunsdale's death (father of Clarence Norman Brunsdale)", "1899", "1899"),
    ("Clarence Norman Brunsdale graduated from Luther College", "1913", "1913"),
    ("Clarence Norman Brunsdale married Carrie Lajord", "August 30, 1925", "August 30, 1925"),
    ("Clarence Norman Brunsdale served in North Dakota State Senate", "1927", "1934"),
    ("Clarence Norman Brunsdale served again in North Dakota State Senate", "1941", "1951"),
    ("Clarence Norman Brunsdale was an alternate delegate to Republican National Convention", "1940", "1940"),
    ("Clarence Norman Brunsdale was a member of Republican National Committee from North Dakota", "1948", "1952"),
    ("Clarence Norman Brunsdale served as Governor of North Dakota", "1951", "1957"),
    ("Clarence Norman Brunsdale was appointed U.S. Senator", "November 19, 1959", "November 19, 1959"),
    ("Clarence Norman Brunsdale's term as U.S. Senator", "November 19, 1959", "August 7, 1960"),
    ("Clarence Norman Brunsdale's death", "January 27, 1978", "January 27, 1978"),
    ("Clarence Norman Brunsdale was buried in Mayville Cemetery", "1978", "1978")
]

Above is the example of how various temporal events are been extracted. You are an assistant specialized in extracted structural temporal events from cintext provided.Your task is to identify and list all relevant events related to the mentioned person's career, along with their start and end dates.
After generating the events lets go through the context again and find all the relevant temporal subjects and examine if that subject or its relvant temporal data present in the events we generated. If not lets extract the event and its temporal data and add it to our final list.
"""

            "Question: extract temporal events in the (Event, Start, End) format from the context below, mention the subject involved at the start of each event sentence, return as a list of tuples\n\n"
            "formatting each event as (Event, Start Date, End Date):\n\nContext: "+" "+"\n\n"+ context + "\n\nEvents:"
        )
        event_extraction_prompts.append(prompt)
        entry_ids.append(entry_id)
        questions.append(question)
        ground_truths_list.append(ground_truths)
        contexts.append(context)

    sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=1024)
    if event_extraction_prompts:
        outputs = llm.generate(event_extraction_prompts, sampling_params)
    else:
        outputs = []

    extracted_events_list = []
    for idx, output in enumerate(outputs):
        response = output.outputs[0].text.strip()
        # print("-----------Level 1 raw output of extracted events-----------")
        # print(response)
        events = parse_events(response)
        normalized_events = normalize_events(events)
        extracted_events_list.append(normalized_events)

    final_answer_prompts = []
    final_entry_map = []
    graphs_dir = "graphs"
    os.makedirs(graphs_dir, exist_ok=True)

    for idx, (entry_id, question, ground_truths, normalized_events) in enumerate(zip(entry_ids, questions, ground_truths_list, extracted_events_list)):
        if not normalized_events:
            results.append({
                'id': entry_id,
                'question': question,
                'generated_answer': "Unable to extract reliable events.",
                'ground_truths': ground_truths,
                'exact_match': 0,
                'f1_score': 0.0
            })
            continue

        temporal_graph = construct_temporal_graph(normalized_events)
        graph_filename = os.path.join(graphs_dir, f'graph_{entry_id}.png')
        # visualize_graph(temporal_graph, graph_filename)

        nodes_data = "\n".join([
            f"Node {n}: {d['event']} ({d['start'].date()} to {d['end'].date()})"
            for n, d in temporal_graph.nodes(data=True)
        ])

        edges_data = "\n".join([
            f"{temporal_graph.nodes[u]['event']} --{d['relation']}--> {temporal_graph.nodes[v]['event']}"
            for u, v, d in temporal_graph.edges(data=True)
        ])
# prompt for answering the test question based on context
        prompt = (
            "Below is some information derived from the provided context. "
            "Do not repeat the raw nodes or edges data in your answer. "
            "Carefully check whether question contains after/before. If after we have to check the start_date of an event which is just after the end_date of the event asked in question. If it is before we have to check end_date of an event which is just before start_date of the event mentioned in question. Also there might me several events which have before/after relation with the event mentioned in the question. As mentioned we have to check an event which is the most nearest one."
            "Conclude the answer using the data provided and after arriving the answer check whether the relation mentioned in Edges data matches with your provided answer to the question. "
            "Simply provide a direct and concise answer to the question. Example: If the answer is "+ "Columbia University"+" then just output "+"Columbia University" +" without any extra text\n\n"
            f"Question: {question}\n\n"
            f"Nodes:\n\n{nodes_data}\n\n"
            f"Edges:\n\n{edges_data}\n\n"
            "Answer:"
        )

        final_answer_prompts.append(prompt)
        final_entry_map.append((entry_id, question, ground_truths, graph_filename))

    if final_answer_prompts:
        final_outputs = llm.generate(final_answer_prompts, sampling_params)
    else:
        final_outputs = []


    idx_count = 0
    for (entry_id, question, ground_truths, graph_filename), output in zip(final_entry_map, final_outputs):
        response = output.outputs[0].text.strip()
        match = re.match(r"^[^\n]+", response)
        if match:
            generated_answer = match.group()
        else:
            generated_answer = response
        em, f1 = calculate_metrics([generated_answer], [ground_truths], tokenizer)
        results.append({
            'id': entry_id,
            'question': question,
            'generated_answer': generated_answer,
            'ground_truths': ground_truths,
            'exact_match': em,
            'f1_score': f1,
            'graph_image': graph_filename
        })
        idx_count += 1

    return results


def main():
    input_json_path = 'test_l3.json'
    output_json_path = 'processed_test_l3_results.json'
    base_model_name = 'meta-llama/Llama-3.1-8b'


    print("Loading dataset...")
    data = load_dataset(input_json_path)
    print(f"Loaded {len(data)} entries.")
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token 

    torch.cuda.empty_cache()
    llm = LLM(model=base_model_name)

    results = []
    predictions = []
    ground_truths_all = []

    batch_size = 32

    print("Processing entries...")
    for i in tqdm(range(0, len(data), batch_size), desc="Processing Entries"):
        batch_entries = data[i:i+batch_size]
        batch_results = process_entries(batch_entries, llm, tokenizer)
        results.extend(batch_results)
        for res in batch_results:
            predictions.append(res['generated_answer'])
            ground_truths_all.append(res['ground_truths'])

    print("Calculating overall metrics...")
    overall_em, overall_f1 = calculate_metrics(predictions, ground_truths_all, tokenizer)
    print(f"Overall Exact Match (EM): {overall_em:.2f}%")
    print(f"Overall F1 Score: {overall_f1:.2f}%")

    print(f"Saving results to {output_json_path}...")
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False, default=str)
    print("Processing completed.")

if __name__ == "__main__":
    main()