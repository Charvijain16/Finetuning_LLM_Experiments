import os
import torch
from datasets import Dataset
import json
from prompt_templates import get_prompt_template
from trl import SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoTokenizer, TrainingArguments
from peft import (
        get_peft_model, 
        LoraConfig
    )
from unsloth import FastLanguageModel
from unsloth import is_bfloat16_supported
import argparse
import json
import re
from pathlib import Path

from utils import categorize_size

# Set up argument parsing
parser = argparse.ArgumentParser(description="Load configuration for the script")
parser.add_argument('--config', type=str, required=True, help="Path to the JSON config file")
args = parser.parse_args()

# Load the config file
with open(args.config, 'r') as f:
    config = json.load(f)
    
os.environ['XFORMERS_MORE_DETAILS'] = '1'

base_dir = Path(config["output_dir_path"])
is_full_finetuning = config["is_full_finetuning"]

model_dir = f"{"full_ft" if is_full_finetuning else "lora_ft"}_{categorize_size(config["model_name"])}_{config["max_seq_length"]//1024}k_{os.path.basename(args.config)}"
result_dir= base_dir / model_dir
result_dir.mkdir(parents=True, exist_ok=True)

prompt= get_prompt_template(config["model_name"])

example="""
Here is an example:

Question:Is mount baker taller than mount st. helens?
Thought:First, need to check more information about Mount Baker.
Action:WikiSearch
Action Input: Mount Baker
Observation: Page: Mount Baker
Summary: Mount Baker is a 10,781 ft (3,286 m) active glacier-covered stratovolcano in the Cascade Volcanic Arc and the North Cascades of Washington in the United States.
Thought:Now, need to check more information about Mount St. Helens.
Action:WikiSearch
Action Input: Mount St. Helens
Observation: Page: Mount St. Helens
Summary: Mount St. Helens is an active stratovolcano located in Skamania County, Washington. The major eruption of May 18, 1980 reduced the elevation of the mountain's summit from 9,677 ft (2,950 m) to 8,363 ft (2,549 m), leaving a 1 mile (1.6 km) wide horseshoe-shaped crater.
Thought:Need to verify the answer by querying over Wikidata.
Action:GetWikidataID
Action Input: Mount Baker, Mount St. Helens
Observation: ['Q594387', 'Q4675']
Thought:With the QIDs, next step is to generate SPARQL query
Action:GenerateSparql
Action Input: Q594387,Q4675
Observation: The possible reason is
 1) The query is syntactically wrong
Thought:To determine the relative heights of Mount Baker and Mount St. Helens using Wikidata.
Action:RunSparql
Action Input: ASK WHERE {{ BIND(wd:Q594387 AS ?baker) BIND(wd:Q4675 AS ?helen) ?baker wdt:P2044 ?bakerElevation . ?helen wdt:P2044 ?helenElevation . FILTER(?bakerElevation > ?helenElevation) }}
Observation: true
Final Answer: Wikipedia_Answer:Yes, Wikidata_Answer: [True]
Assistant Response: Mount Baker, with an elevation of 10,781 ft (3,286 m), is taller than Mount St. Helens, which has an elevation of 8,363 ft (2,549 m) after the 1980 eruption. Both Wikipedia and Wikidata confirm this information. Mount Baker is a 10,781 ft (3,286 m) active glacier-covered stratovolcano in the Cascade Volcanic Arc and the North Cascades of Washington in the United States. Mount St. Helens is an active stratovolcano located in Skamania County, Washington. The major eruption of May 18, 1980 reduced the elevation of the mountain's summit from 9,677 ft (2,950 m) to 8,363 ft (2,549 m), leaving a 1 mile (1.6 km) wide horseshoe-shaped crater."""

system_instruction = """Given the question, your task is to find the answer using both Wikipedia and Wikidata Databases.If you found the answer using Wikipedia Article you need to verify it with Wikidata, even if you do not find an answer with Wikpedia, first make sure to look up on different relevant wikipedia articles. If you still cannot find with wikipedia, try with Wikidata as well.
When Wikipedia gives no answer or SPARQL query gives no result, you are allowed to use relevant keywords for finding QIDs to generate the SPARQL query.
Your immediate steps include finding relevant wikipedia articles summary to find the answer using tools provided, find Keywords that are the QIDS from the Wikidata using Wikipedia Page title. \nUse these QIDs to generate the SPARQL query using available tools.\nWikidata Answers are the observation after executing the SPARQL query.\n
Also do not check the wikipedia page manually to answer the questions.
You have access to the following tools:\n\nWikiSearch:Useful to find relevant Wikipedia article given the Action Input. Do not use this tool with same Action Input.\nGetWikidataID:useful to get QIDs given the Action Input. Do not use this tool with same Action Input.\nGenerateSparql:useful to get Squall query given the Action Input. Do not use this tool with same Action Input.\nRunSparql:useful to run a query on wikibase to get results. Do not use this tool with same Action Input.\nWikiSearchSummary:useful to find the answer on wikipedia article given the Action Input if WikiSearch Tool doesnt provide any answer!. Do not use this tool with same Action Input.\nGetLabel:useful to get the label for the wikidata QID. Do not use this tool with same Action Input!.
Once you have the Wikipedia Answer and Wikidata Answer or either of them after trying, always follow the specific format to output the final answer -
Final Answer: Wikipedia_Answer : Wikipedia Answer, Wikidata_Answer : Wikidata Answer ,
Assistant Response: Extended Answer that contains your reasoning, proof and final answer, please keep this descriptive.
Please do not use same Action Input to the tools, If no answer is found even after multiple tries using wikidata but found answer with wikipedia return and vice versa
Wikipedia_Answer : Answer, Wikidata_Answer : None
"""

format_instruction = """Use the following format:
Question: the input question for which you must provide a natural language answer
Thought: you should always think about what to do
Action: the action to take, should be one of WikiSearch, GetWikidataID, GenerateSparql, RunSparql, WikiSearchSummary, GetLabel
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Always use the following format for the Final Answer -
Final Answer: Wikipedia_Answer : , Wikidata_Answer : ,"""

def get_device_map() -> str:
    return 'cuda' if torch.cuda.is_available() else 'cpu'

device = config.get("device")# "auto" #get_device_map()


### Load dataset
f1 = open(config.get("train_data_path"))
train_dataset = json.load(f1)

full_prompt_dataset = []
for i, entry in enumerate(train_dataset):
  text = prompt.format(instruction=f"{system_instruction if config["use_sys_instruction"] else ""}{example if config["use_example"] else ""}{format_instruction if config["use_format_instruction"] else ""}", question=entry["question"], output=entry["assistant_reponse"])
  full_prompt_dataset.append(text)

training_dataset = { "text" : full_prompt_dataset, }

def gen(dataset):
  for i in dataset:
    yield {"text": i}

train_ds = Dataset.from_generator(gen, gen_kwargs={"dataset": training_dataset["text"]})

use_unsloth = "unsloth" in config["model_name"]

if use_unsloth:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = config["model_name"],
        max_seq_length = config["max_seq_length"],
        dtype = config["dtype"],
        load_in_4bit =  config["load_in_4bit"],
    )
else:
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"], trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        trust_remote_code=True,
        cache_dir='',
        use_cache = False,
        device_map = device,
    )

if not is_full_finetuning:
    peft_config = LoraConfig(
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        r=config["lora_r"],
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["k_proj", "q_proj", "v_proj", "up_proj", "down_proj", "gate_proj", "o_proj"],
        modules_to_save=["embed_tokens", "input_layernorm", "post_attention_layernorm", "norm"],
        use_rslora=False
    )
else:
    peft_config = None

if not is_full_finetuning:
    if use_unsloth:
        lora_model = FastLanguageModel.get_peft_model(
            model,
            use_gradient_checkpointing = "unsloth",
            random_state = 3407,
            **peft_config
        )
    else:
        lora_model = get_peft_model(model, peft_config)


def use_bf16(support:bool):
    if support:
        return is_bfloat16_supported()
    else: 
        return not is_bfloat16_supported()

training_arguments = TrainingArguments(
    output_dir=result_dir,
    seed = 3407,
    auto_find_batch_size=config["auto_find_batch_size"],
    per_device_train_batch_size=config["per_device_train_batch_size"],
    gradient_accumulation_steps=config["gradient_accumulation_steps"],
    optim=config["optimizer"],
    save_steps=config["save_steps"],
    logging_steps=config["logging_steps"],
    learning_rate=config["learning_rate"],
    fp16 = not use_bf16(config["bf16"]),
    bf16 = use_bf16(config["bf16"]),
    max_grad_norm=config["max_grad_norm"],
    # num_train_epochs = 1, # Set this for 1 full training run.
    max_steps=config["max_steps"],
    weight_decay = config["weight_decay"],
    warmup_steps = config["warmup_steps"],
    #warmup_ratio=warmup_ratio,
    group_by_length=True,
    lr_scheduler_type= config["lr_scheduler_type"],
    gradient_checkpointing=config["gradient_checkpointing"],
    gradient_checkpointing_kwargs =  config["gradient_checkpointing_kwargs"],#must be false for DDP
    report_to=config["report_to"],
)

# Trainer
trainer = SFTTrainer(
    model=model if is_full_finetuning else lora_model,
    train_dataset=train_ds,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=config["max_seq_length"],
    tokenizer=tokenizer,
    dataset_num_proc = 2,
    args=training_arguments,
    packing = config["packing"], # Can make training 5x faster for short sequences.
)

print("###################################################")

print("Show memory stats before training")
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")
print("###################################################")

trainer_stats = trainer.train()

print("###################################################")
print("Show memory stats after training")
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory         /max_memory*100, 3)
lora_percentage = round(used_memory_for_lora/max_memory*100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")
print("###################################################")

if not is_full_finetuning:
    folder_name = "final_adapters"
    output_path = result_dir / folder_name
    output_path.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(output_path)
    trainer.tokenizer.save_pretrained(output_path)
        
# from transformers import AutoModelForCausalLM
# from peft import PeftModel

# base_model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
# peft_model_id = "alignment-handbook/zephyr-7b-sft-lora"
# model = PeftModel.from_pretrained(base_model, peft_model_id)
# model.merge_and_unload()