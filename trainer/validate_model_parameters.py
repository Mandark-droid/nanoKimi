import torch
from transformers import PretrainedConfig, PreTrainedModel
import math

# Define the scaled-down DeepseekV3Config (from previous response)
class DeepseekV3Config(PretrainedConfig):
    model_type = "kimi_k2"
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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

# Mock DeepseekV3ForCausalLM class (since actual implementation is unavailable)
class DeepseekV3ForCausalLM(PreTrainedModel):
    config_class = DeepseekV3Config
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        # Mock the model structure for parameter counting and forward pass
        self.word_embeddings = torch.nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = torch.nn.ModuleList([
            torch.nn.ModuleDict({
                'attention': torch.nn.Linear(config.hidden_size, config.hidden_size),
                'moe': torch.nn.ModuleDict({
                    'shared_expert': torch.nn.Sequential(
                        torch.nn.Linear(config.hidden_size, config.moe_intermediate_size),
                        torch.nn.SiLU(),
                        torch.nn.Linear(config.moe_intermediate_size, config.hidden_size)
                    ),
                    'experts': torch.nn.ModuleList([
                        torch.nn.Sequential(
                            torch.nn.Linear(config.hidden_size, config.moe_intermediate_size),
                            torch.nn.SiLU(),
                            torch.nn.Linear(config.moe_intermediate_size, config.hidden_size)
                        ) for _ in range(config.n_routed_experts)
                    ]),
                    'gate': torch.nn.Linear(config.hidden_size, config.n_routed_experts)
                }),
                'norm': torch.nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
            }) for _ in range(config.num_hidden_layers)
        ])
        self.norm = torch.nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        # If tie_word_embeddings=True, reuse word_embeddings for output
        self.lm_head = self.word_embeddings if config.tie_word_embeddings else torch.nn.Linear(config.hidden_size, config.vocab_size)

    def forward(self, input_ids, attention_mask=None):
        x = self.word_embeddings(input_ids)
        for layer in self.layers:
            # Simplified forward pass: attention + MoE
            x = layer['attention'](x)
            gate_logits = layer['moe']['gate'](x)
            topk_experts = torch.topk(gate_logits, self.config.num_experts_per_tok, dim=-1).indices
            expert_outputs = torch.zeros_like(x)
            for i in range(self.config.num_experts_per_tok):
                expert_idx = topk_experts[..., i]
                for b in range(x.size(0)):
                    for s in range(x.size(1)):
                        expert_output = layer['moe']['experts'][expert_idx[b, s]](x[b, s])
                        expert_outputs[b, s] += expert_output
            shared_output = layer['moe']['shared_expert'](x)
            x = x + (expert_outputs + shared_output) / (self.config.num_experts_per_tok + 1)
            x = layer['norm'](x)
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits

def count_total_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    return total_params

def count_active_parameters(config, total_params):
    # Calculate active parameters for one forward pass in an MoE model
    # Active parameters include: embeddings, attention, norms, gate, shared expert, and top-k experts
    embedding_params = config.vocab_size * config.hidden_size
    if not config.tie_word_embeddings:
        embedding_params += config.vocab_size * config.hidden_size  # Add lm_head if not tied

    attention_params_per_layer = config.hidden_size * config.hidden_size * 2  # Q, K, V, and output projection
    norm_params_per_layer = config.hidden_size * 2  # LayerNorm weights and biases
    gate_params_per_layer = config.hidden_size * config.n_routed_experts
    shared_expert_params_per_layer = config.hidden_size * config.moe_intermediate_size * 2  # Two linear layers
    expert_params_per_layer = config.hidden_size * config.moe_intermediate_size * 2 * config.num_experts_per_tok  # Only top-k experts

    layer_params = (attention_params_per_layer + norm_params_per_layer + gate_params_per_layer +
                    shared_expert_params_per_layer + expert_params_per_layer)
    total_layer_params = layer_params * config.num_hidden_layers
    final_norm_params = config.hidden_size * 2

    active_params = embedding_params + total_layer_params + final_norm_params
    return active_params

def validate_forward_pass(model, config):
    # Create a sample input
    batch_size = 2
    seq_length = 16
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length)).to(torch.long)
    attention_mask = torch.ones(batch_size, seq_length).to(torch.long)

    # Perform forward pass
    model.eval()
    with torch.no_grad():
        try:
            outputs = model(input_ids, attention_mask=attention_mask)
            # Check output shape
            expected_shape = (batch_size, seq_length, config.vocab_size)
            assert outputs.shape == expected_shape, f"Expected output shape {expected_shape}, got {outputs.shape}"
            print(f"Forward pass successful! Output shape: {outputs.shape}")
            return True
        except Exception as e:
            print(f"Forward pass failed: {str(e)}")
            return False

def main():
    # Initialize model
    config = MODEL_CONFIG
    model = DeepseekV3ForCausalLM(config)
    print(model)
    # Count total parameters
    total_params = count_total_parameters(model)
    print(f"Total parameters: {total_params:,}")

    # Estimate active parameters for one forward pass
    active_params = count_active_parameters(config, total_params)
    print(f"Active parameters (approx.): {active_params:,}")

    # Validate forward pass
    forward_pass_success = validate_forward_pass(model, config)
    if forward_pass_success:
        print("Model validation completed successfully.")
    else:
        print("Model validation failed.")

if __name__ == "__main__":
    main()