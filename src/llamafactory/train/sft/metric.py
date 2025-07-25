# Copyright 2025 HuggingFace Inc., THUDM, and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library and the THUDM's ChatGLM implementation.
# https://github.com/huggingface/transformers/blob/v4.40.0/examples/pytorch/summarization/run_summarization.py
# https://github.com/THUDM/ChatGLM-6B/blob/main/ptuning/main.py
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

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch
from transformers.utils import is_jieba_available, is_nltk_available

from ...extras.constants import IGNORE_INDEX
from ...extras.misc import numpify
from ...extras.packages import is_rouge_available


if TYPE_CHECKING:
    from transformers import EvalPrediction, PreTrainedTokenizer


if is_jieba_available():
    import jieba  # type: ignore


if is_nltk_available():
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu  # type: ignore


if is_rouge_available():
    from rouge_chinese import Rouge  # type: ignore


def eval_logit_processor(logits: "torch.Tensor", _: "torch.Tensor") -> "torch.Tensor":
    r"""Compute the token with the largest likelihood to reduce memory footprint."""
    if isinstance(logits, (list, tuple)):
        if logits[0].dim() == 3:  # (batch_size, seq_len, vocab_size)
            logits = logits[0]
        else:  # moe models have aux loss
            logits = logits[1]

    if logits.dim() != 3:
        raise ValueError("Cannot process the logits.")

    return torch.argmax(logits, dim=-1)


@dataclass
class ComputeAccuracy:
    r"""Compute accuracy and support `batch_eval_metrics`."""

    def _dump(self) -> Optional[dict[str, float]]:
        result = None
        if hasattr(self, "score_dict"):
            result = {k: float(np.mean(v)) for k, v in self.score_dict.items()}

        self.score_dict = {"accuracy": []}
        return result

    def __post_init__(self):
        self._dump()

    def __call__(self, eval_preds: "EvalPrediction", compute_result: bool = True) -> Optional[dict[str, float]]:
        preds, labels = numpify(eval_preds.predictions), numpify(eval_preds.label_ids)
        for i in range(len(preds)):
            pred, label = preds[i, :-1], labels[i, 1:]
            label_mask = label != IGNORE_INDEX
            self.score_dict["accuracy"].append(np.mean(pred[label_mask] == label[label_mask]))

        if compute_result:
            return self._dump()


@dataclass
class ComputeSimilarity:
    r"""Compute text similarity scores and support `batch_eval_metrics`.

    Wraps the tokenizer into metric functions, used in CustomSeq2SeqTrainer.
    """

    tokenizer: "PreTrainedTokenizer"

    def _dump(self) -> Optional[dict[str, float]]:
        result = None
        if hasattr(self, "score_dict"):
            result = {k: float(np.mean(v)) for k, v in self.score_dict.items()}

        self.score_dict = {"rouge-1": [], "rouge-2": [], "rouge-l": [], "bleu-4": []}
        return result

    def __post_init__(self):
        self._dump()

    def __call__(self, eval_preds: "EvalPrediction", compute_result: bool = True) -> Optional[dict[str, float]]:
        preds, labels = numpify(eval_preds.predictions), numpify(eval_preds.label_ids)

        preds = np.where(preds != IGNORE_INDEX, preds, self.tokenizer.pad_token_id)
        labels = np.where(labels != IGNORE_INDEX, labels, self.tokenizer.pad_token_id)

        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        for pred, label in zip(decoded_preds, decoded_labels):
            hypothesis = list(jieba.cut(pred))
            reference = list(jieba.cut(label))

            if len(" ".join(hypothesis).split()) == 0 or len(" ".join(reference).split()) == 0:
                result = {"rouge-1": {"f": 0.0}, "rouge-2": {"f": 0.0}, "rouge-l": {"f": 0.0}}
            else:
                rouge = Rouge()
                scores = rouge.get_scores(" ".join(hypothesis), " ".join(reference))
                result = scores[0]

            for k, v in result.items():
                self.score_dict[k].append(round(v["f"] * 100, 4))

            bleu_score = sentence_bleu([list(label)], list(pred), smoothing_function=SmoothingFunction().method3)
            self.score_dict["bleu-4"].append(round(bleu_score * 100, 4))

        if compute_result:
            return self._dump()


@dataclass
class ComputeDetailedMetrics:
    r"""Compute detailed metrics during training and support `batch_eval_metrics`.
    
    Supports entropy, perplexity, and tool calling specific metrics.
    """
    
    tokenizer: Optional["PreTrainedTokenizer"] = None
    compute_entropy: bool = True
    compute_perplexity: bool = True
    compute_reasoning_metrics: bool = False
    compute_tool_call_metrics: bool = False
    entropy_k: int = 10
    
    def _dump(self) -> Optional[dict[str, float]]:
        result = None
        if hasattr(self, "score_dict"):
            result = {k: float(np.mean(v)) for k, v in self.score_dict.items()}

        # Initialize metrics based on configuration
        metrics = []
        
        # Base metrics
        if self.compute_entropy:
            metrics.append("entropy")
            metrics.extend(["top1_prob", "top2_prob", "top5_prob", "top10_prob"])

        if self.compute_perplexity:
            metrics.append("perplexity")

        # Tool calling specific metrics
        if self.compute_reasoning_metrics:
            reasoning_metrics = ["reasoning_entropy", "reasoning_perplexity"]
            reasoning_metrics.extend(["reasoning_top1_prob", "reasoning_top2_prob", "reasoning_top5_prob", "reasoning_top10_prob"])
            metrics.extend([m for m in reasoning_metrics 
                           if (m.endswith("_entropy") and self.compute_entropy) or 
                              (m.endswith("_perplexity") and self.compute_perplexity) or
                              (m.endswith("_prob") and self.compute_entropy)])
                              
        if self.compute_tool_call_metrics:
            tool_call_metrics = ["tool_call_entropy", "tool_call_perplexity"]
            tool_call_metrics.extend(["tool_call_top1_prob", "tool_call_top2_prob", "tool_call_top5_prob", "tool_call_top10_prob"])
            metrics.extend([m for m in tool_call_metrics
                           if (m.endswith("_entropy") and self.compute_entropy) or 
                              (m.endswith("_perplexity") and self.compute_perplexity) or
                              (m.endswith("_prob") and self.compute_entropy)])
        
        self.score_dict = {metric: [] for metric in metrics}
        return result

    def __post_init__(self):
        self._dump()

    def __call__(self, eval_preds: "EvalPrediction", compute_result: bool = True) -> Optional[dict[str, float]]:
        logits, labels = eval_preds.predictions, eval_preds.label_ids
        
        # Handle different logits formats (tuple/list from some models)
        if isinstance(logits, (list, tuple)):
            if logits[0].dim() == 3:  # (batch_size, seq_len, vocab_size)
                logits = logits[0]
            else:  # moe models have aux loss
                logits = logits[1]
        
        if logits.ndim != 3:
            raise ValueError("Cannot process the logits for entropy computation.")
        
        # Convert to torch tensors if needed
        if not isinstance(logits, torch.Tensor):
            logits = torch.from_numpy(logits)
        if not isinstance(labels, torch.Tensor):
            labels = torch.from_numpy(labels)
        
        # Check if we have preprocessed masks available
        preprocessed_masks = {}
        if hasattr(eval_preds, 'inputs') and eval_preds.inputs is not None:
            for mask_name in ['reasoning_mask', 'tool_call_mask']:
                if mask_name in eval_preds.inputs:
                    mask_data = eval_preds.inputs[mask_name]
                    if not isinstance(mask_data, torch.Tensor):
                        mask_data = torch.from_numpy(mask_data)
                    preprocessed_masks[mask_name] = mask_data
        
        # Calculate entropy for each sample
        for i in range(len(logits)):
            sample_logits = logits[i, :-1]  # exclude last token
            sample_labels = labels[i, 1:]   # exclude first token (shift for next token prediction)
            
            # Only compute entropy for non-ignored positions
            label_mask = sample_labels != IGNORE_INDEX
            if label_mask.sum() == 0:
                continue
            
            # Get logits for valid positions
            valid_logits = sample_logits[label_mask]  # (valid_seq_len, vocab_size)
            
            if preprocessed_masks:
                # Use preprocessed masks (more efficient and accurate)
                self._compute_tool_calling_metrics_with_masks(
                    sample_logits, sample_labels, label_mask, i, preprocessed_masks
                )
            elif self.tokenizer is not None:
                # Fallback to runtime classification
                self._compute_tool_calling_metrics(sample_logits, sample_labels, label_mask, i, logits, labels)
            else:
                # Only base metrics if no tokenizer available
                valid_labels = sample_labels[label_mask]
                self._compute_regular_entropy(valid_logits, valid_labels)

        if compute_result:
            return self._dump()
            
    def _compute_regular_entropy(self, valid_logits: torch.Tensor, valid_labels: torch.Tensor = None):
        """Compute regular entropy metrics."""
        if valid_logits.size(0) == 0:
            return  # No valid tokens to compute entropy
            
        # Calculate probabilities
        probs = torch.softmax(valid_logits, dim=-1)
        
        # Get top-k logits and probabilities
        k = min(self.entropy_k, valid_logits.size(-1))
        top_k_probs, _ = torch.topk(probs, k=k, dim=-1)
        
        # Calculate Binary Shannon entropy: -p*log2(p) - (1-p)*log2(1-p)
        # where p is the probability of the most likely token
        p = top_k_probs[:, 0]  # Top-1 probability
        
        # For numerical stability, handle edge cases
        # Binary entropy is 0 when p=0 or p=1, and max (1.0) when p=0.5
        # Use a small epsilon to avoid log(0)
        epsilon = 1e-8
        p_clamped = torch.clamp(p, min=epsilon, max=1.0 - epsilon)
        
        # Binary Shannon entropy formula
        # H(p) = -p*log2(p) - (1-p)*log2(1-p)
        binary_entropy = -(p_clamped * torch.log2(p_clamped) + (1 - p_clamped) * torch.log2(1 - p_clamped))
        
        # Handle edge cases where p is very close to 0 or 1
        # In these cases, entropy should be close to 0
        binary_entropy = torch.where(
            (p < epsilon) | (p > 1.0 - epsilon),
            torch.zeros_like(binary_entropy),
            binary_entropy
        )
        
        # Average entropy across sequence
        mean_entropy = binary_entropy.mean()
        
        # Ensure the result is valid before appending
        if torch.isfinite(mean_entropy):
            avg_entropy = mean_entropy.item()
            self.score_dict["entropy"].append(avg_entropy)
            
            # Calculate perplexity based on cross-entropy loss (if labels available)
            if "perplexity" in self.score_dict and valid_labels is not None:
                # Cross-entropy: -log(p_true_token)
                log_probs = torch.log_softmax(valid_logits, dim=-1)
                # Gather log probs for true labels
                cross_entropy = -log_probs.gather(1, valid_labels.unsqueeze(1)).squeeze(1)
                
                # Clamp cross entropy to avoid extreme values
                cross_entropy = torch.clamp(cross_entropy, max=100.0)  # Prevent overflow in exp
                mean_cross_entropy = cross_entropy.mean()
                
                if torch.isfinite(mean_cross_entropy):
                    perplexity = torch.exp(mean_cross_entropy).item()
                    # Clamp perplexity to reasonable range
                    perplexity = min(perplexity, 1e6)  # Avoid extremely large values
                    self.score_dict["perplexity"].append(perplexity)
            
            # Calculate accumulated probabilities for top 1, 2, 5, 10
            for top_n in [1, 2, 5, 10]:
                if top_n <= k:
                    # Sum probabilities for top-n tokens
                    top_n_prob = top_k_probs[:, :top_n].sum(dim=-1).mean().item()
                    key = f"top{top_n}_prob"
                    if key in self.score_dict:
                        self.score_dict[key].append(top_n_prob)
    
    def _compute_tool_calling_metrics(
        self, 
        sample_logits: torch.Tensor, 
        sample_labels: torch.Tensor, 
        label_mask: torch.Tensor,
        sample_idx: int,
        all_logits: torch.Tensor,
        all_labels: torch.Tensor
    ):
        """Compute tool calling specific metrics."""
        # Decode the input to classify tokens
        input_ids = all_labels[sample_idx]  # Use labels as they contain the target sequence
        text = self.tokenizer.decode(input_ids, skip_special_tokens=False)
        
        # Classify tokens
        token_classification = self._classify_tokens(text, input_ids)
        
        # Ensure classification matches sequence length
        seq_len = min(len(token_classification), sample_logits.size(0), sample_labels.size(0))
        classification = token_classification[:seq_len]
        
        # Create masks for different token types
        reasoning_mask = torch.tensor([c == 'reasoning' for c in classification], 
                                    device=sample_logits.device, dtype=torch.bool)
        tool_call_mask = torch.tensor([c == 'tool_call' for c in classification], 
                                    device=sample_logits.device, dtype=torch.bool)  
        
        # Combine with label mask
        reasoning_mask = reasoning_mask & label_mask[:seq_len]
        tool_call_mask = tool_call_mask & label_mask[:seq_len]
        
        # Compute metrics for each token type
        self._compute_entropy_for_mask(sample_logits[:seq_len], reasoning_mask, "reasoning")
        self._compute_entropy_for_mask(sample_logits[:seq_len], tool_call_mask, "tool_call")
        
        # Also compute overall entropy
        overall_valid_mask = label_mask[:seq_len]
        if overall_valid_mask.sum() > 0:
            valid_logits = sample_logits[:seq_len][overall_valid_mask]
            self._compute_entropy_for_logits(valid_logits, "")
    
    def _compute_entropy_for_mask(self, logits: torch.Tensor, mask: torch.Tensor, prefix: str):
        """Compute entropy for tokens matching the given mask."""
        if mask.sum() > 0:
            masked_logits = logits[mask]
            self._compute_entropy_for_logits(masked_logits, prefix)
    
    def _compute_entropy_for_logits(self, logits: torch.Tensor, prefix: str, labels: torch.Tensor = None):
        """Compute entropy for given logits."""
        if logits.size(0) == 0:
            return  # No valid tokens to compute entropy
            
        # Calculate probabilities
        probs = torch.softmax(logits, dim=-1)
        
        # Get top-k logits and probabilities
        k = min(self.entropy_k, logits.size(-1))
        top_k_probs, _ = torch.topk(probs, k=k, dim=-1)
        
        # Calculate Binary Shannon entropy: -p*log2(p) - (1-p)*log2(1-p)
        # where p is the probability of the most likely token
        p = top_k_probs[:, 0]  # Top-1 probability
        
        # For numerical stability, handle edge cases
        epsilon = 1e-8
        p_clamped = torch.clamp(p, min=epsilon, max=1.0 - epsilon)
        
        # Binary Shannon entropy formula
        binary_entropy = -(p_clamped * torch.log2(p_clamped) + (1 - p_clamped) * torch.log2(1 - p_clamped))
        
        # Handle edge cases where p is very close to 0 or 1
        binary_entropy = torch.where(
            (p < epsilon) | (p > 1.0 - epsilon),
            torch.zeros_like(binary_entropy),
            binary_entropy
        )
        
        entropy = binary_entropy.mean()
        
        entropy_key = f"{prefix}_entropy" if prefix else "entropy"
        perplexity_key = f"{prefix}_perplexity" if prefix else "perplexity"
        
        # Only log metrics that are enabled and exist in score_dict
        if torch.isfinite(entropy):
            if entropy_key in self.score_dict:
                self.score_dict[entropy_key].append(entropy.item())
                
            # Calculate perplexity based on cross-entropy loss (if labels available)
            if perplexity_key in self.score_dict and labels is not None:
                # Cross-entropy: -log(p_true_token)
                log_probs = torch.log_softmax(logits, dim=-1)
                cross_entropy = -log_probs.gather(1, labels.unsqueeze(1)).squeeze(1)
                
                # Clamp cross entropy to avoid extreme values
                cross_entropy = torch.clamp(cross_entropy, max=100.0)
                mean_cross_entropy = cross_entropy.mean()
                
                if torch.isfinite(mean_cross_entropy):
                    perplexity = torch.exp(mean_cross_entropy).item()
                    perplexity = min(perplexity, 1e6)  # Avoid extremely large values
                    self.score_dict[perplexity_key].append(perplexity)
                
            # Calculate accumulated probabilities for top 1, 2, 5, 10
            for top_n in [1, 2, 5, 10]:
                if top_n <= k:
                    # Sum probabilities for top-n tokens
                    top_n_prob = top_k_probs[:, :top_n].sum(dim=-1).mean().item()
                    key = f"{prefix}_top{top_n}_prob" if prefix else f"top{top_n}_prob"
                    if key in self.score_dict:
                        self.score_dict[key].append(top_n_prob)
    
    def _classify_tokens(self, text: str, input_ids: torch.Tensor) -> list[str]:
        """Classify each token in the sequence using simplified two-level approach."""
        # Use the same logic as classify_tokens_for_tool_calling but adapted for runtime classification
        from ...data.collator import classify_tokens_for_tool_calling
        
        # Get token classification masks 
        token_masks = classify_tokens_for_tool_calling(text, input_ids, self.tokenizer)
        
        # Convert masks to classification strings
        reasoning_mask = token_masks['reasoning_mask']
        tool_call_mask = token_masks['tool_call_mask']
        
        classification = []
        for i in range(len(reasoning_mask)):
            if reasoning_mask[i]:
                classification.append('reasoning')
            elif tool_call_mask[i]:
                classification.append('tool_call')
            else:
                classification.append('special')  # Neither reasoning nor tool_call (e.g., special tokens)
        
        return classification
    
    def _compute_tool_calling_metrics_with_masks(
        self, 
        sample_logits: torch.Tensor, 
        sample_labels: torch.Tensor, 
        label_mask: torch.Tensor,
        sample_idx: int,
        preprocessed_masks: dict[str, torch.Tensor]
    ):
        """Compute tool calling specific metrics using preprocessed masks."""
        # Validate that required masks are available
        required_masks = ['reasoning_mask', 'tool_call_mask']
        for mask_name in required_masks:
            if mask_name not in preprocessed_masks:
                raise ValueError(f"Required mask '{mask_name}' not found in preprocessed masks. "
                               f"Available masks: {list(preprocessed_masks.keys())}")
        
        seq_len = min(sample_logits.size(0), sample_labels.size(0))
        
        # Extract masks for this sample
        reasoning_mask = preprocessed_masks['reasoning_mask']
        tool_call_mask = preprocessed_masks['tool_call_mask']
        
        # Validate batch dimensions
        if sample_idx >= reasoning_mask.size(0):
            raise ValueError(f"Sample index {sample_idx} out of bounds for mask batch size {reasoning_mask.size(0)}")
        
        # Shift masks by 1 position to align with labels (next token prediction)
        # reasoning_mask[sample_idx, :-1] corresponds to sample_labels[sample_idx, 1:]
        if reasoning_mask.size(1) <= 1:
            raise ValueError(f"Mask sequence length {reasoning_mask.size(1)} too short for next-token prediction")
            
        reasoning_mask = reasoning_mask[sample_idx, :-1][:seq_len]
        tool_call_mask = tool_call_mask[sample_idx, :-1][:seq_len]  
        
        # Ensure masks are on the same device (important for distributed training)
        reasoning_mask = reasoning_mask.to(sample_logits.device)
        tool_call_mask = tool_call_mask.to(sample_logits.device)
        
        # Combine with label mask
        reasoning_mask = reasoning_mask & label_mask[:seq_len]
        tool_call_mask = tool_call_mask & label_mask[:seq_len]
        
        # Get labels for masked positions (for perplexity calculation)
        valid_labels = sample_labels[:seq_len]
        
        # Compute metrics for each token type
        if reasoning_mask.sum() > 0:
            reasoning_logits = sample_logits[:seq_len][reasoning_mask]
            reasoning_labels = valid_labels[reasoning_mask]
            self._compute_entropy_for_logits(reasoning_logits, "reasoning", reasoning_labels)
            
        if tool_call_mask.sum() > 0:
            tool_call_logits = sample_logits[:seq_len][tool_call_mask]
            tool_call_labels = valid_labels[tool_call_mask]
            self._compute_entropy_for_logits(tool_call_logits, "tool_call", tool_call_labels)
        
        # Also compute overall entropy
        overall_valid_mask = label_mask[:seq_len]
        if overall_valid_mask.sum() > 0:
            valid_logits = sample_logits[:seq_len][overall_valid_mask]
            valid_labels_masked = valid_labels[overall_valid_mask]
            self._compute_entropy_for_logits(valid_logits, "", valid_labels_masked)
