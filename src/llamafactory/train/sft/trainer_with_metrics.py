# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn.functional as F
from typing import Dict, Any, Union
from typing_extensions import override

from ...extras.constants import IGNORE_INDEX
from .trainer import CustomSeq2SeqTrainer


class DetailedMetricsSeq2SeqTrainer(CustomSeq2SeqTrainer):
    r"""
    Extends CustomSeq2SeqTrainer to compute detailed metrics during both training and evaluation.
    """

    def __init__(
        self,
        compute_entropy: bool = True,
        compute_perplexity: bool = True,
        compute_reasoning_metrics: bool = False,
        compute_tool_call_metrics: bool = False,
        entropy_k: int = 10,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.compute_entropy = compute_entropy
        self.compute_perplexity = compute_perplexity
        self.compute_reasoning_metrics = compute_reasoning_metrics
        self.compute_tool_call_metrics = compute_tool_call_metrics
        self.entropy_k = entropy_k

    @override
    def compute_loss(self, model, inputs, return_outputs=False, *args, **kwargs):
        """
        Compute loss and detailed metrics efficiently using single forward pass.
        """
        labels = inputs.get("labels")
        
        # Single forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        if labels is not None and logits is not None:
            # Compute loss manually to match what super() would do
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten for loss computation
            flat_logits = shift_logits.view(-1, shift_logits.size(-1))
            flat_labels = shift_labels.view(-1)
            
            # Compute cross-entropy loss
            loss = F.cross_entropy(flat_logits, flat_labels, ignore_index=IGNORE_INDEX)
            
            # Compute metrics if enabled and we have required inputs
            if self._should_compute_metrics(inputs):
                metrics = self._compute_metrics(shift_logits, shift_labels, loss, inputs)
                self._log_metrics(metrics)
        else:
            loss = outputs.loss if hasattr(outputs, 'loss') else None
        
        return (loss, outputs) if return_outputs else loss
    
    def _should_compute_metrics(self, inputs: Dict[str, Any]) -> bool:
        """Check if we should compute metrics."""
        if not any([self.compute_entropy, self.compute_perplexity, 
                   self.compute_reasoning_metrics, self.compute_tool_call_metrics]):
            return False
            
        if (self.compute_reasoning_metrics or self.compute_tool_call_metrics):
            return all(key in inputs for key in ['reasoning_mask', 'tool_call_mask'])
        return True
    
    def _compute_metrics(
        self, 
        shift_logits: torch.Tensor,  # (batch_size, seq_len-1, vocab_size)
        shift_labels: torch.Tensor,  # (batch_size, seq_len-1)
        loss: torch.Tensor,
        inputs: Dict[str, Any]
    ) -> Dict[str, float]:
        """Compute all metrics efficiently."""
        # Valid positions mask
        valid_mask = shift_labels != IGNORE_INDEX
        if valid_mask.sum() == 0:
            return {}
        
        # Flatten and extract valid positions
        flat_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_labels = shift_labels.view(-1)
        flat_valid_mask = valid_mask.view(-1)
        
        valid_logits = flat_logits[flat_valid_mask]
        valid_labels = flat_labels[flat_valid_mask]
        
        # Compute probabilities once
        with torch.no_grad():
            probs = torch.softmax(valid_logits, dim=-1)
            k = min(self.entropy_k, valid_logits.size(-1))
            top_k_probs, _ = torch.topk(probs, k=k, dim=-1)
        
        metrics = {}
        
        # Overall metrics
        if self.compute_entropy:
            metrics['entropy'] = self._compute_entropy(top_k_probs[:, 0])
        
        if self.compute_perplexity:
            metrics['perplexity'] = min(torch.exp(loss).item(), 1e6)
        
        # Top-k probabilities
        for top_n in [1, 2, 5, 10]:
            if top_n <= k:
                metrics[f'top{top_n}_prob'] = top_k_probs[:, :top_n].sum(dim=-1).mean().item()
        
        # Masked metrics for reasoning and tool_call
        if self.compute_reasoning_metrics or self.compute_tool_call_metrics:
            masked_metrics = self._compute_masked_metrics(
                inputs, valid_mask, flat_valid_mask, top_k_probs, loss, k
            )
            metrics.update(masked_metrics)
        
        return metrics
    
    def _compute_entropy(self, top1_probs: torch.Tensor) -> float:
        """Compute binary Shannon entropy efficiently."""
        epsilon = 1e-8
        p = torch.clamp(top1_probs, min=epsilon, max=1.0 - epsilon)
        entropy = -(p * torch.log2(p) + (1 - p) * torch.log2(1 - p))
        entropy = torch.where((top1_probs < epsilon) | (top1_probs > 1.0 - epsilon),
                             torch.zeros_like(entropy), entropy)
        return entropy.mean().item()
    
    def _compute_masked_metrics(
        self,
        inputs: Dict[str, Any],
        valid_mask: torch.Tensor,  # (batch_size, seq_len-1)
        flat_valid_mask: torch.Tensor,  # (batch_size*(seq_len-1),)
        all_top_k_probs: torch.Tensor,  # (num_valid, k)
        loss: torch.Tensor,
        k: int
    ) -> Dict[str, float]:
        """Compute metrics for reasoning and tool_call tokens."""
        metrics = {}
        
        # Get shifted masks to align with labels
        reasoning_mask = inputs['reasoning_mask'][:, :-1].to(valid_mask.device)
        tool_call_mask = inputs['tool_call_mask'][:, :-1].to(valid_mask.device)
        
        # Combine with valid mask and flatten
        reasoning_valid = (reasoning_mask & valid_mask).view(-1)
        tool_call_valid = (tool_call_mask & valid_mask).view(-1)
        
        # Map to valid_logits indices
        valid_indices = torch.cumsum(flat_valid_mask, dim=0) - 1
        
        # Reasoning metrics
        if self.compute_reasoning_metrics:
            reasoning_indices = valid_indices[reasoning_valid]
            if reasoning_indices.numel() > 0:
                reasoning_probs = all_top_k_probs[reasoning_indices]
                
                if self.compute_entropy:
                    metrics['reasoning_entropy'] = self._compute_entropy(reasoning_probs[:, 0])
                if self.compute_perplexity:
                    metrics['reasoning_perplexity'] = min(torch.exp(loss).item(), 1e6)
                
                for top_n in [1, 2, 5, 10]:
                    if top_n <= k:
                        metrics[f'reasoning_top{top_n}_prob'] = reasoning_probs[:, :top_n].sum(dim=-1).mean().item()
        
        # Tool call metrics
        if self.compute_tool_call_metrics:
            tool_call_indices = valid_indices[tool_call_valid]
            if tool_call_indices.numel() > 0:
                tool_call_probs = all_top_k_probs[tool_call_indices]
                
                if self.compute_entropy:
                    metrics['tool_call_entropy'] = self._compute_entropy(tool_call_probs[:, 0])
                if self.compute_perplexity:
                    metrics['tool_call_perplexity'] = min(torch.exp(loss).item(), 1e6)
                
                for top_n in [1, 2, 5, 10]:
                    if top_n <= k:
                        metrics[f'tool_call_top{top_n}_prob'] = tool_call_probs[:, :top_n].sum(dim=-1).mean().item()
        
        return metrics
    
    def _log_metrics(self, metrics: Dict[str, float]):
        """Log metrics with train/eval prefix."""
        if metrics:
            prefix = "train_" if self.model.training else "eval_"
            prefixed_metrics = {f"{prefix}{k}": v for k, v in metrics.items()}
            self.log(prefixed_metrics)