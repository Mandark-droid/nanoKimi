import os
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from huggingface_hub.errors import RepositoryNotFoundError
import wandb
from datasets import load_dataset
from transformers import DeepseekV3ForCausalLM, DeepseekV3Config, AutoTokenizer, AutoModelForCausalLM
from transformers.integrations import WandbCallback
from huggingface_hub import login, HfApi, create_repo
import logging
from tqdm.auto import tqdm
from collections import Counter
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

# Environment setup
os.environ["WANDB_PROJECT"] = "nano-kimi-training"
os.environ["WANDB_LOG_MODEL"] = "checkpoint"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # RTX 4090
HF_TOKEN = os.getenv("HF_TOKEN", "api_here")
login(token=HF_TOKEN)
# Initialize W&B run
wandb.init(
    project="nano-kimi-training",
    config={},  # Will update with training_config and MODEL_CONFIG later
    name="nano-kimi-moe-training",
)
# Device and dtype setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_type = 'cuda' if 'cuda' in str(device) else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_bf16_supported() else 'float16'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == 'cuda' else nullcontext()

# Model configuration
# Model configuration
MODEL_CONFIG = DeepseekV3Config(
    architectures=["DeepseekV3ForCausalLM"],
    attention_bias=False,
    attention_dropout=0.0,
    auto_map={
        "AutoConfig": "configuration_deepseek.DeepseekV3Config",
        "AutoModel": "modeling_deepseek.DeepseekV3Model",
        "AutoModelForCausalLM": "modeling_deepseek.DeepseekV3ForCausalLM"
    },
    aux_loss_alpha=0.001,
    bos_token_id=163584,
    eos_token_id=163585,
    first_k_dense_replace=1,
    hidden_act="silu",
    hidden_size=512,
    initializer_range=0.02,
    intermediate_size=768,
    kv_lora_rank=128,
    max_position_embeddings=1024,
    model_type="kimi_k2",
    moe_intermediate_size=512,
    moe_layer_freq=1,
    n_group=1,
    n_routed_experts=8,
    n_shared_experts=1,
    norm_topk_prob=True,
    num_attention_heads=8,
    num_experts_per_tok=2,
    num_hidden_layers=8,
    num_key_value_heads=2,
    num_nextn_predict_layers=0,
    pretraining_tp=1,
    q_lora_rank=384,
    qk_nope_head_dim=32,
    qk_rope_head_dim=32,
    quantization_config={
        "activation_scheme": "dynamic",
        "fmt": "e4m3",
        "quant_method": "fp8",
        "weight_block_size": [128, 128]
    },
    rms_norm_eps=1e-6,
    rope_theta=1000000,
    rope_scaling=None,
    routed_scaling_factor=1.0,
    scoring_func="sigmoid",
    seq_aux=True,
    tie_word_embeddings=False,
    topk_group=1,
    topk_method="noaux_tc",
    torch_dtype="bfloat16",
    transformers_version="4.52.4",
    use_cache=True,
    v_head_dim=64,
    vocab_size=163840
)

HF_MODEL_REPO = "kshitijthakkar/loggenix-nano-kimi-moe-0.3B-A0.1B-e5-lr5e4-b1-4096"
DATASET_PATH = "kshitijthakkar/loggenix_moe_mcs_v0"
MIN_TOKENS = 50
MAX_CONTEXT_LENGTH = 4096
batch_size = 1  # Changed from 4 to 1 for 3060 revert back to 4 for 4090
block_size = 4096  # Reduced from 4096 to save VRAM
num_train_epochs = 5  # Increased from 3
total_sequences = 699462  # Adjust after filtering

# Split-to-expert mapping
SPLIT_MAPPING = {
    'follow_up': 1, 'code_': 2, 'open_domain_qa': 3, 'rag': 4, 'text_modification': 5,
    'text_extraction': 6, 'rc': 7, 'creative_content': 8, 'struct2text_flow': 9,
    'mcq': 10, 'text_classification': 11, 'brain_teaser': 12, 'fs_cot_flow': 13,
    'analytical_reasoning': 14, 'fermi': 15
}

# Generation prompts
GENERATION_PROMPTS = {
    'follow_up': "Provide a follow-up response to: 'Can you explain neural networks?'",
    'code_': "Write a Python function to compute factorial.",
    'open_domain_qa': "What is the capital of France?",
    'rag': "Retrieve and summarize the latest research on large language models.",
    'text_modification': "Rephrase this sentence to sound more professional: 'Hey, I need that report fast.'",
    'text_extraction': "Extract all the dates from this paragraph: 'John met Lisa on Jan 5, and they went to Paris on Feb 14.'",
    'rc': "Read the paragraph and answer: 'Why did the boy cry?' - Paragraph: 'The boy cried because he lost his toy.'",
    'creative_content': "Write a short poem about space exploration.",
    'struct2text_flow': "Convert this JSON to a natural language sentence: {'user': 'Alice', 'action': 'logged in', 'time': '5PM'}",
    'mcq': "Choose the correct option: What is the boiling point of water? A) 90°C B) 100°C C) 110°C D) 120°C",
    'text_classification': "Classify the sentiment: 'I absolutely loved the new Batman movie!'",
    'brain_teaser': "Solve this riddle: What has keys but can't open locks?",
    'fs_cot_flow': "Answer this question using step-by-step reasoning: If it takes 3 painters 6 hours to paint a wall, how long would 2 painters take?",
    'analytical_reasoning': "If A is older than B and B is older than C, who is the oldest?",
    'fermi': "Estimate: How many tennis balls can fit inside a standard-sized airplane cabin?"
}

# Data preprocessing
# Data preprocessing
def filter_and_clip_tokens(example):
    token_count = example["total_token_count"]
    return MIN_TOKENS <= token_count <= MAX_CONTEXT_LENGTH

def process(example):
    ids = example["encoded_text"]
    if not ids or len(ids) == 0:
        logger.warning(f"Empty encoded_text for example: {example}")
        return None
    if len(ids) != example["total_token_count"]:
        logger.warning(f"Mismatch in token count: {example['total_token_count']} vs {len(ids)}")
        return None
    split_id = SPLIT_MAPPING.get(example["split_name"], 16)
    return {'ids': ids, 'len': len(ids), 'split_id': split_id}

# Load and preprocess dataset
try:
    logger.info("Loading dataset in non-streaming mode...")
    dataset = load_dataset(DATASET_PATH, streaming=False)
    streaming = False
except MemoryError:
    logger.warning("Insufficient memory for non-streaming mode. Falling back to streaming mode...")
    dataset = load_dataset(DATASET_PATH, streaming=True)
    streaming = True

train_dataset = dataset["train"].filter(filter_and_clip_tokens).shuffle(seed=42)
eval_dataset = dataset["test"].filter(filter_and_clip_tokens).shuffle(seed=42)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("moonshotai/Moonlight-16B-A3B-Instruct", trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# Verify encoded_text compatibility
logger.info("Verifying encoded_text compatibility...")
for example in dataset["train"].take(5):
    ids = example["encoded_text"]
    decoded = tokenizer.decode(ids, skip_special_tokens=False)
    logger.info(f"Decoded: {decoded[:100]}... | Total token count: {example['total_token_count']} | Actual length: {len(ids)}")

# Tokenize and save to memory-mapped files
AscendingNumeric = np.uint16 if tokenizer.vocab_size < 65536 else np.uint32

# Check for existing files or download from Hugging Face
if all(os.path.exists(f) for f in ["train.bin", "validation.bin", "train_split_ids.bin", "validation_split_ids.bin"]):
    logger.info("Reusing existing preprocessed files")
else:
    from huggingface_hub import hf_hub_download
    try:
        for file in ["train.bin", "validation.bin", "train_split_ids.bin", "validation_split_ids.bin"]:
            if not os.path.exists(file):
                logger.info(f"Downloading {file} from Hugging Face...")
                hf_hub_download(repo_id=HF_MODEL_REPO, filename=f"preprocessed/{file}", repo_type="model", token=HF_TOKEN, local_dir=".")
    except Exception as e:
        logger.info(f"Preprocessed files not found on Hugging Face: {e}. Creating new files...")

        # Preprocessing for train.bin
        if not os.path.exists("train.bin"):
            logger.info("Creating train.bin and train_split_ids.bin...")
            tokenized = train_dataset.map(
                process,
                remove_columns=['formatted_text', 'encoded_text', 'total_token_count', 'messages', 'split_name']
            ).filter(lambda x: x is not None)
            arr_len = 0
            for item in tqdm(tokenized, desc="Calculating train array length"):
                arr_len += item['len']
            arr = np.memmap("train.bin", dtype=AscendingNumeric, mode='w+', shape=(arr_len,))
            split_ids = np.memmap("train_split_ids.bin", dtype=np.uint8, mode='w+', shape=(arr_len,))
            idx = 0
            for batch in tqdm(tokenized, desc="Writing train.bin"):
                arr_batch = np.array(batch['ids'], dtype=AscendingNumeric)
                arr[idx:idx + len(arr_batch)] = arr_batch
                split_ids[idx:idx + len(arr_batch)] = batch['split_id']
                idx += len(arr_batch)
                wandb.log({"preprocessing_step": "writing_train_bin", "progress": idx / arr_len}, commit=False)
            arr.flush()
            split_ids.flush()

        # Preprocessing for validation.bin
        if not os.path.exists("validation.bin"):
            logger.info("Creating validation.bin and validation_split_ids.bin...")
            tokenized = eval_dataset.map(
                process,
                remove_columns=['formatted_text', 'encoded_text', 'total_token_count', 'messages', 'split_name']
            ).filter(lambda x: x is not None)
            arr_len = 0
            for item in tqdm(tokenized, desc="Calculating validation array length"):
                arr_len += item['len']
            arr = np.memmap("validation.bin", dtype=AscendingNumeric, mode='w+', shape=(arr_len,))
            split_ids = np.memmap("validation_split_ids.bin", dtype=np.uint8, mode='w+', shape=(arr_len,))
            idx = 0
            for batch in tqdm(tokenized, desc="Writing validation.bin"):
                arr_batch = np.array(batch['ids'], dtype=AscendingNumeric)
                arr[idx:idx + len(arr_batch)] = arr_batch
                split_ids[idx:idx + len(arr_batch)] = batch['split_id']
                idx += len(arr_batch)
                wandb.log({"preprocessing_step": "writing_validation_bin", "progress": idx / arr_len}, commit=False)
            arr.flush()
            split_ids.flush()

        # Upload to Hugging Face
        api = HfApi()
        for file in ["train.bin", "validation.bin", "train_split_ids.bin", "validation_split_ids.bin"]:
            api.upload_file(
                path_or_fileobj=file,
                path_in_repo=f"preprocessed/{file}",
                repo_id=HF_MODEL_REPO,
                repo_type="model",
                token=HF_TOKEN
            )
            logger.info(f"Uploaded {file} to Hugging Face")

# Update total_sequences
logger.info("Estimating total_sequences...")
total_train_tokens = sum(ex["total_token_count"] for ex in tqdm(dataset["train"].filter(filter_and_clip_tokens), desc="Counting train tokens"))
total_sequences = total_train_tokens // block_size
logger.info(f"Updated total_sequences: {total_sequences}")

# Batch retrieval function
def get_batch(split):
    data = np.memmap(f'{split}.bin', dtype=AscendingNumeric, mode='r')
    split_ids = np.memmap(f'{split}_split_ids.bin', dtype=np.uint8, mode='r')
    max_start_idx = len(data) - block_size
    if max_start_idx <= 0:
        raise ValueError(f"Dataset too small for block_size {block_size}")
    ix = torch.randint(max_start_idx, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i + block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i + 1:i + 1 + block_size]).astype(np.int64)) for i in ix])
    split_id = torch.tensor([split_ids[i] for i in ix], dtype=torch.long)
    x, y, split_id = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True), split_id.to(device, non_blocking=True)
    return x, y, split_id


def log_expert_usage(outputs, global_step):
    """Fixed expert usage logging with proper error handling and debugging"""
    if outputs.router_logits is not None and len(outputs.router_logits) > 0:
        try:
            # Debug: Print shapes of router logits
            # print(f"Number of router logit layers: {len(outputs.router_logits)}")
            # for i, logits in enumerate(outputs.router_logits):
            # print(f"Layer {i} router logits shape: {logits.shape}")

            # Stack router logits from all layers
            stacked_logits = torch.stack(outputs.router_logits, dim=0)
            # print(f"Stacked logits shape: {stacked_logits.shape}")

            # Check if we have the expected dimensions
            if stacked_logits.dim() < 3:
                print(f"WARNING: Unexpected stacked logits dimensions: {stacked_logits.dim()}")
                return

            # Apply softmax to get probabilities over the last dimension (experts)
            expert_probs = torch.softmax(stacked_logits, dim=-1)
            # print(f"Expert probs shape: {expert_probs.shape}")

            # Average across all dimensions except the last (expert) dimension
            # This should give us [num_experts]
            if expert_probs.dim() == 4:  # [num_layers, batch, seq_len, num_experts]
                expert_usage = expert_probs.mean(dim=(0, 1, 2))
            elif expert_probs.dim() == 3:  # [num_layers, batch, num_experts]
                expert_usage = expert_probs.mean(dim=(0, 1))
            elif expert_probs.dim() == 2:  # [batch, num_experts]
                expert_usage = expert_probs.mean(dim=0)
            else:
                print(f"Unexpected expert_probs dimensions: {expert_probs.dim()}")
                return

            # print(f"Expert usage shape: {expert_usage.shape}")
            # print(f"Expert usage sum: {expert_usage.sum().item()}")
            # print(f"Expert usage values: {expert_usage}")

            # Check if expert_usage is actually a tensor with multiple elements
            if expert_usage.dim() == 0:
                print("ERROR: expert_usage is a scalar (0-d tensor)")
                print("This suggests router logits don't have expert dimension")
                return

            if expert_usage.shape[0] == 0:
                print("ERROR: expert_usage has 0 elements")
                return

            # Log usage for all experts
            expert_usage_dict = {}
            for i in range(expert_usage.shape[0]):
                expert_usage_dict[f"expert_usage_{i}"] = expert_usage[i].item()

            wandb.log(expert_usage_dict, step=global_step)
            # Log summary statistics about expert usage
            # Calculate load balance metrics
            expert_usage_dict.update({
                "expert_usage_entropy": -torch.sum(expert_usage * torch.log(expert_usage + 1e-10)).item(),
                "expert_usage_std": expert_usage.std().item(),
                "expert_usage_max": expert_usage.max().item(),
                "expert_usage_min": expert_usage.min().item(),
                "active_experts": (expert_usage > 0.01).sum().item(),  # Count experts with >1% usage
                "effective_num_experts": torch.exp(-torch.sum(expert_usage * torch.log(expert_usage + 1e-10))).item()
            })

            wandb.log(expert_usage_dict, step=global_step)

            # Log top experts if we have enough experts
            if expert_usage.shape[0] >= 5:
                top_experts = torch.topk(expert_usage, k=5)

                # Log top expert indices and values individually
                top_expert_dict = {}
                for i, (idx, val) in enumerate(zip(top_experts.indices.tolist(), top_experts.values.tolist())):
                    top_expert_dict[f"top_expert_{i}_index"] = idx
                    top_expert_dict[f"top_expert_{i}_usage"] = val

                #wandb.log(top_expert_dict, step=global_step)

                # Alternative: Log as a table if you want to see the data together
                wandb.log({
                    "top_experts_table": wandb.Table(
                        columns=["rank", "expert_index", "usage"],
                        data=[[i, idx, val] for i, (idx, val) in enumerate(zip(top_experts.indices.tolist(), top_experts.values.tolist()))]
                    )
                }, step=global_step)

        except Exception as e:
            # print(f"Error in expert usage logging: {e}")
            # print(f"Router logits shapes: {[logits.shape for logits in outputs.router_logits]}")
            # Log basic info without failing
            wandb.log({"expert_logging_error": str(e)}, step=global_step)

# def compute_mean_token_accuracy(model, eval_dataloader, device):
#     model.eval()
#     total_correct = 0
#     total_tokens = 0
#
#     with torch.no_grad():
#         for batch in eval_dataloader:
#             input_ids = batch["input_ids"].to(device)
#             attention_mask = batch["attention_mask"].to(device)
#             labels = batch["labels"].to(device)
#
#             outputs = model(input_ids=input_ids, attention_mask=attention_mask)
#             logits = outputs.logits  # (batch, seq_len, vocab)
#
#             predictions = torch.argmax(logits, dim=-1)  # (batch, seq_len)
#             mask = labels != -100  # Only evaluate non-masked positions
#
#             correct = (predictions == labels) & mask
#             total_correct += correct.sum().item()
#             total_tokens += mask.sum().item()
#
#     model.train()  # Set model back to training mode
#
#     if total_tokens == 0:
#         return 0.0
#     mean_token_accuracy=total_correct / total_tokens
#     return mean_token_accuracy

# Custom callback
class CustomWandbCallback(WandbCallback):
    def __init__(self, tokenizer, generation_prompts):
        super().__init__()
        self.tokenizer = tokenizer
        self.generation_prompts = generation_prompts

    def on_evaluate(self, args, state, control, model, metrics, **kwargs):
        global_step = kwargs.get('global_step', 0)
        super().on_evaluate(args, state, control, metrics=metrics, **kwargs)
        wandb.log(metrics, step=global_step)
        model.eval()
        generated_outputs = {}
        for split, prompt in self.generation_prompts.items():
            if not prompt or not prompt.strip():
                logger.warning(f"Skipping invalid prompt for split: {split} (prompt: '{prompt}')")
                continue
            logger.debug(f"Tokenizing prompt for split {split}: '{prompt}'")
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=MAX_CONTEXT_LENGTH,
                truncation=True,
                add_special_tokens=True,
                return_attention_mask=True
            )
            logger.debug(f"Input IDs shape: {inputs['input_ids'].shape}, Attention Mask shape: {inputs['attention_mask'].shape}")
            if inputs['input_ids'].shape[1] == 0 or inputs['attention_mask'].sum() == 0:
                logger.warning(f"Invalid input for split {split}, using fallback prompt")
                prompt = "Hello world"
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    max_length=MAX_CONTEXT_LENGTH,
                    truncation=True,
                    add_special_tokens=True,
                    return_attention_mask=True
                )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            try:
                torch.cuda.empty_cache()
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=100,  # Increased from 25
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        return_dict_in_generate=True,
                        use_cache=False
                    )
                generated_text = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
                generated_outputs[f"{split}_generated"] = generated_text
                memory = model.get_memory_footprint() / 1e6
                logger.info(f"Memory footprint: {memory:,.1f} MB")
            except Exception as e:
                logger.error(f"Generation failed for split {split}: {str(e)}")
                continue
        wandb.log(generated_outputs, step=global_step)
        with open("generated_outputs.txt", "w", encoding="utf-8") as f:
            for key, value in generated_outputs.items():
                f.write(f"{key}: {value}\n")
        try:
            api = HfApi()
            try:
                api.repo_info(repo_id=HF_MODEL_REPO, repo_type="model", token=HF_TOKEN)
            except RepositoryNotFoundError:
                logger.info(f"Repository {HF_MODEL_REPO} not found, creating it...")
                api.create_repo(repo_id=HF_MODEL_REPO, token=HF_TOKEN, repo_type="model", private=True)
            api.upload_file(
                path_or_fileobj="generated_outputs.txt",
                path_in_repo=f"eval/step_{global_step}/generated_outputs.txt",
                repo_id=HF_MODEL_REPO,
                repo_type="model",
                token=HF_TOKEN
            )
            os.remove("generated_outputs.txt")
        except Exception as e:
            logger.error(f"Failed to upload to Hugging Face: {str(e)}")

# Training configuration
training_config = {
    "learning_rate": 5e-4,  # Adjusted from 7e-4
    "num_train_epochs": 5,
    "warmup_ratio": 0.1,
    "min_lr": 1e-5,  # Adjusted from 1e-6
    "eval_interval": 5000,
    "gradient_accumulation_steps": 1,  # Reduced from 8
    "batch_size": batch_size,
    "block_size": block_size
}

## Update W&B config with training and model configurations

wandb.config.update({**training_config, **MODEL_CONFIG.to_dict()})

# Initialize model
try:
    logger.info("Loading model from Hugging Face checkpoint...")
    # model = AutoModelForCausalLM.from_pretrained(
    #     HF_MODEL_REPO,
    #     subfolder="checkpoints/epoch_1_step_280000",
    #     trust_remote_code=True,
    #     torch_dtype=torch.bfloat16
    # ).to(device)
    model = DeepseekV3ForCausalLM.from_pretrained(
        "kshitijthakkar/loggenix-nano-kimi-moe-0.3B-A0.1B-e5-lr5e4-b1-4096",
        torch_dtype=torch.float16,  # Adjust dtype if needed
        device_map="auto"  # Optional: for GPU/CPU placement
    )
except Exception as e:
    logger.warning(f"Failed to load checkpoint: {e}. Initializing new model...")
    model = DeepseekV3ForCausalLM(MODEL_CONFIG).to(device)

model.to(torch.bfloat16)
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

# Print model info
total_params = sum(p.numel() for p in model.parameters())
logger.info(f"Total parameters: {total_params:,}")
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
logger.info(f"Trainable parameters: {trainable_params:,}")
memory = model.get_memory_footprint() / 1e6
logger.info(f"Memory footprint: {memory:,.1f} MB")
logger.info(f"Current VRAM: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB")
logger.info(f"Peak VRAM: {torch.cuda.max_memory_allocated(device) / 1e9:.2f} GB")

# Optimizer and scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=training_config["learning_rate"], betas=(0.9, 0.95), weight_decay=0.1)
total_steps = (total_sequences // batch_size) * num_train_epochs
warmup_steps = int(training_config["warmup_ratio"] * total_steps)
#scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=total_steps // num_train_epochs, T_mult=1, eta_min=training_config["min_lr"])
scheduler_warmup = torch.optim.lr_scheduler.LinearLR(optimizer, total_iters=warmup_steps)
scheduler_decay = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps,
                                                             eta_min=training_config["min_lr"])
scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[scheduler_warmup, scheduler_decay],
                                                  milestones=[warmup_steps])
scaler = torch.amp.GradScaler('cuda', enabled=(dtype == 'float16'))

# Loss estimation
def estimate_loss():
    out = {}
    model.eval()
    with torch.no_grad():
        for split in ['train', 'validation']:
            losses = torch.zeros(200)
            total_correct = 0
            total_tokens = 0

            for k in range(200):
                X, Y, split_id = get_batch(split)
                with ctx:
                    outputs = model(input_ids=X, labels=Y, split_id=split_id)
                    if outputs.loss is None:
                        raise ValueError("Loss is None in evaluation")
                    losses[k] = outputs.loss.item()

                    # Calculate token accuracy for this batch
                    logits = outputs.logits  # (batch, seq_len, vocab)
                    predictions = torch.argmax(logits, dim=-1)  # (batch, seq_len)
                    mask = Y != -100  # Only evaluate non-masked positions

                    correct = (predictions == Y) & mask
                    total_correct += correct.sum().item()
                    total_tokens += mask.sum().item()

            # Calculate mean token accuracy for this split
            mean_token_accuracy = total_correct / total_tokens if total_tokens > 0 else 0.0

            out[split] = losses.mean().item()
            out[f'{split}_accuracy'] = mean_token_accuracy

    model.train()
    return out

# Training loop
best_val_loss = float('inf')
best_model_path = "best_model_params.pt"
patience = 30  # Increased from 20
no_improvement_count = 0
train_loss_list = []
validation_loss_list = []
current_best_checkpoint_path = None

for epoch in range(num_train_epochs):
    for iter_num in tqdm(range(0, total_steps // num_train_epochs), desc=f"Epoch {epoch + 1}"):
        global_step = epoch * (total_steps // num_train_epochs) + iter_num
        if iter_num % training_config["eval_interval"] == 0 and iter_num > 0:
            losses = estimate_loss()
            #mean_token_accuracy = compute_mean_token_accuracy(model, eval_dataloader, device)
            train_accuracy = losses['train_accuracy']
            val_accuracy = losses['validation_accuracy']
            logger.info(f"Epoch {epoch + 1}, Step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['validation']:.4f}")
            train_loss_list.append(losses['train'])
            validation_loss_list.append(losses['validation'])
            wandb.log({
                "train_loss": losses['train'],
                "val_loss": losses['validation'],
                "iteration": global_step,
                "learning_rate": optimizer.param_groups[0]['lr'],
                "gradient_norm": sum(p.grad.norm(2).item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5,
                "train_accuracy": train_accuracy,
                "val_accuracy": val_accuracy

            })
            logger.info(f"Current VRAM: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB")
            logger.info(f"Peak VRAM: {torch.cuda.max_memory_allocated(device) / 1e9:.2f} GB")
            callback = CustomWandbCallback(tokenizer, GENERATION_PROMPTS)
            callback.on_evaluate(None, None, None, model, metrics=losses, global_step=global_step)

            if losses['validation'] < best_val_loss:
                best_val_loss = losses['validation']
                torch.save(model.state_dict(), best_model_path)
                new_checkpoint_path = f"checkpoints/epoch_{epoch + 1}_step_{iter_num}/pytorch_model.bin"
                api = HfApi()
                if current_best_checkpoint_path is not None:
                    try:
                        api.delete_file(
                            path_in_repo=current_best_checkpoint_path,
                            repo_id=HF_MODEL_REPO,
                            repo_type="model",
                            token=HF_TOKEN
                        )
                        logger.info(f"Deleted previous checkpoint: {current_best_checkpoint_path}")
                    except Exception as e:
                        logger.warning(f"Failed to delete previous checkpoint {current_best_checkpoint_path}: {e}")
                api.upload_file(
                    path_or_fileobj=best_model_path,
                    path_in_repo=new_checkpoint_path,
                    repo_id=HF_MODEL_REPO,
                    repo_type="model",
                    token=HF_TOKEN
                )
                logger.info(f"Uploaded new best checkpoint: {new_checkpoint_path}")
                current_best_checkpoint_path = new_checkpoint_path
                os.remove(best_model_path)
                no_improvement_count = 0
            else:
                no_improvement_count += 1
                if no_improvement_count >= patience:
                    logger.info("Early stopping triggered.")
                    break

        try:
            X, y, split_id = get_batch("train")
            with ctx:
                outputs = model(input_ids=X, labels=y, split_id=split_id)
                if outputs.router_logits is not None and len(outputs.router_logits) > 0:
                    log_expert_usage(outputs, global_step)
                if outputs.loss is None:
                    logger.error(f"X shape: {X.shape}, y shape: {y.shape}, split_id shape: {split_id.shape}")
                    raise ValueError("Loss is None in training")
                loss = outputs.loss / training_config["gradient_accumulation_steps"]
                scaler.scale(loss).backward()
        except RuntimeError as e:
            logger.error(f"Training failed: {e}")
            if "out of memory" in str(e).lower():
                logger.info("OOM detected. Try further reducing batch size or sequence length.")
            raise

        if (iter_num + 1) % training_config["gradient_accumulation_steps"] == 0 or (iter_num + 1) == (total_steps // num_train_epochs):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        scheduler.step()

# Save final model
if os.path.exists(best_model_path):
    model.load_state_dict(torch.load(best_model_path, map_location=device))
model.push_to_hub(HF_MODEL_REPO)
tokenizer.push_to_hub(HF_MODEL_REPO)
wandb.finish()

# Clean up local storage
for file in ['train.bin', 'validation.bin', 'train_split_ids.bin', 'validation_split_ids.bin', 'best_model_params.pt']:
    if os.path.exists(file):
        os.remove(file)