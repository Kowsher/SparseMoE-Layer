import torch
import torch.nn as nn
import torch.nn.functional as F

class SparseMoeBlock(nn.Module):
    """
    It's faster since it formulates MoE operations
    in terms of block-sparse operations to accommodate imbalanced
    assignments of tokens to experts, whereas standard MoE either
    (1) drop tokens at the cost of reduced performance or (2) set
    capacity factor to number of experts and thus waste computation
    and memory on padding.
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.intermediate_size
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok

        # gating
        self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=False)

        self.experts = nn.ModuleList([nn.Sequential(
            nn.Linear(self.hidden_dim, self.ffn_dim),
            nn.GELU(),
            nn.Linear(self.ffn_dim, self.hidden_dim)
        ) for _ in range(self.num_experts)])

        # Jitter parameters
        self.jitter_noise = config.router_jitter_noise

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
       
        if self.training and self.jitter_noise > 0:
            hidden_states *= torch.empty_like(hidden_states).uniform_(1.0 - self.jitter_noise, 1.0 + self.jitter_noise)
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be solicited
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

            # However `index_add_` only supports torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states


import torch
import torch.nn as nn
import torch.nn.functional as F

class SequenceSparseMoeBlock(nn.Module):
    """
    Sequence-level MoE: Routing is determined at the sequence level instead of at the token level.
    Each sequence in the batch will decide which experts will be used.
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.intermediate_size
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_seq  # Now routing per sequence, not per token

        # Gating mechanism for sequence-level routing
        self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=False)

        self.experts = nn.ModuleList([nn.Sequential(
            nn.Linear(self.hidden_dim, self.ffn_dim),
            nn.GELU(),
            nn.Linear(self.ffn_dim, self.hidden_dim)
        ) for _ in range(self.num_experts)])

        # Jitter parameters
        self.jitter_noise = config.router_jitter_noise

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape

        # Compute sequence representation, e.g., by averaging over tokens
        # Here we are using the mean of all tokens in the sequence to represent the sequence
        sequence_representation = hidden_states.mean(dim=1)  # Shape: (batch_size, hidden_dim)
        
        if self.training and self.jitter_noise > 0:
            sequence_representation *= torch.empty_like(sequence_representation).uniform_(
                1.0 - self.jitter_noise, 1.0 + self.jitter_noise
            )

        # Gating to determine expert probabilities for each sequence
        # router_logits: (batch_size, num_experts)
        router_logits = self.gate(sequence_representation)

        # Get routing weights and selected experts for each sequence
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(sequence_representation.dtype)

        # Initialize final hidden states
        final_hidden_states = torch.zeros(
            (batch_size, sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # One hot encode the selected experts to create an expert mask
        # This will be used to easily index which expert will be used for each sequence
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        # Loop over all available experts in the model and perform computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, seq_indices = torch.where(expert_mask[expert_idx])

            if len(seq_indices) == 0:
                continue  # Skip if no sequence is assigned to this expert

            # Select the sequences assigned to the current expert
            current_hidden_states = hidden_states[seq_indices]  # Shape: (num_sequences, sequence_length, hidden_dim)
            current_hidden_states = current_hidden_states.reshape(-1, hidden_dim)

            # Apply expert transformation
            expert_output = expert_layer(current_hidden_states)
            expert_output = expert_output.reshape(-1, sequence_length, hidden_dim)

            # Weight the output using the routing weights
            expert_output *= routing_weights[seq_indices, idx].unsqueeze(-1).unsqueeze(-1)

            # Add to final hidden states
            final_hidden_states[seq_indices] += expert_output.to(hidden_states.dtype)

        return final_hidden_states

