import torch
import torch.nn as nn
from typing import List, Any
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from peft import LoraConfig, get_peft_model, PeftModel, PeftConfig
import os
import numpy as np
import logging

logger = logging.getLogger(__name__)

class GeneralLLMWrapper(nn.Module):
    """
    General wrapper for any HuggingFace LLM (Qwen, Llama, Mistral, Phi, ...), supporting PEFT/LoRA/quantization.
    - state: string describing the current state/problem
    - act: generates a text action (with randomness)
    - forward: computes loss for each transition in each trajectory (loss = LLM loss on prompt: state + '<answer>' + action)
    - supports saving/loading PEFT/quantized models
    """
    def __init__(self, model_name_or_path: str, peft_config: dict = None, quant_config: dict = None, generation_config: dict = None, device: str = None):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.generation_config = generation_config or {
            'max_new_tokens': 32,
            'temperature': 1.2,
            'do_sample': True,
            'top_p': 0.95,
            'top_k': 50,
            'pad_token_id': 0,
            'eos_token_id': 2
        }
        self.is_peft = False
        self.is_quant = False
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Detect if model_name_or_path is a PEFT/LoRA/quantized model
        try:
            if os.path.exists(model_name_or_path) and os.path.exists(os.path.join(model_name_or_path, 'adapter_config.json')):
                self.is_peft = True
        except Exception:
            pass
        # Load model
        if self.is_peft:
            # Load PEFT model directly
            self.model = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float32), model_name_or_path)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float32)
            # Apply LoRA/PEFT if config provided
            if peft_config is not None:
                lora_config = LoraConfig(**peft_config, task_type="CAUSAL_LM")
                self.model = get_peft_model(self.model, lora_config)
                self.is_peft = True
            # Quantization (if needed, not implemented here for simplicity)
            # ...
        self.model.to(self.device)
        self.model.train()

    def act(self, state: str, deterministic: bool = False) -> str:
        """
        Generate a text action given a string state.
        Args:
            state: string describing the current state/problem
            deterministic: if True, use greedy decoding; else, sample with high temperature
        Returns:
            Generated text (action)
        """
        prompt = state.strip()
        gen_cfg = self.generation_config.copy()
        if deterministic:
            gen_cfg['do_sample'] = False
            gen_cfg['temperature'] = 0.
        input_ids = self.tokenizer([prompt], return_tensors='pt').input_ids.to(self.device)
        with torch.no_grad():
            output_ids = self.model.generate(input_ids, **gen_cfg)
        # Only return the generated part after the prompt
        gen_text = self.tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)

        # remove <step> and </step>
        if gen_text.startswith(prompt):
            return gen_text
        else:
            return prompt + gen_text

    def forward(self, trajectories: List[Any]) -> List[List[torch.Tensor]]:
        """
        Compute loss for each transition in each trajectory.
        Args:
            trajectories: list of trajectory objects, each with .states (list of str), .actions (list of str)
        Returns:
            List of lists of loss tensors (losses[i][j] = loss for j-th transition in i-th trajectory)
        """
        all_losses = []
        for traj in trajectories:
            traj_losses = []
            for state, action in zip(traj.states, traj.actions):
                prompt = state.strip() + str(action)
                inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
                labels = inputs['input_ids'].clone()
                outputs = self.model(**inputs, labels=labels)
                loss = outputs.loss  # scalar
                traj_losses.append(loss)
            all_losses.append(traj_losses)
        return all_losses

    def save(self, path: str):
        if self.is_peft:
            self.model.save_pretrained(path, save_adapter=True)
        else:
            self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def load(self, path: str):
        if os.path.exists(os.path.join(path, 'adapter_config.json')):
            self.model = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32), path)
            self.is_peft = True
        else:
            self.model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32)
            self.is_peft = False
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model.to(self.device)


'''
import os
import torch
from typing import List, Optional
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig
)
from peft import PeftModel, LoraConfig, get_peft_model
from safetensors.torch import save_model

class Trajectory:
    """Mock trajectory class for type hinting and structure."""
    def __init__(self, states: List[str], actions: List[str]):
        self.states = states
        self.actions = actions

class LLMPolicy:
    def __init__(
        self,
        model_name: str,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        use_lora: bool = False,
        lora_rank: int = 8,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        max_length: int = 2048,
        device: Optional[str] = None,
    ):
        """
        Initialize the LLM policy.
        
        Args:
            model_name: Hugging Face model identifier or path
            load_in_4bit: Load model in 4-bit quantization
            load_in_8bit: Load model in 8-bit quantization
            use_lora: Apply LoRA for parameter-efficient training
            lora_rank: LoRA rank
            lora_alpha: LoRA alpha
            lora_dropout: LoRA dropout
            max_length: Maximum token length for input sequences
            device: Device to load the model on (e.g., "cuda", "cpu")
        """
        self.model_name = model_name
        self.load_in_4bit = load_in_4bit
        self.load_in_8bit = load_in_8bit
        self.use_lora = use_lora
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._load_model_and_tokenizer()
        self._configure_lora()

    def _load_model_and_tokenizer(self):
        """Load model and tokenizer with optional quantization."""
        bnb_config = None
        if self.load_in_4bit or self.load_in_8bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=self.load_in_4bit,
                load_in_8bit=self.load_in_8bit,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float32,
            )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, padding_side="right"
        )
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float32,
        )
        self.model.eval()

    def _configure_lora(self):
        """Apply LoRA configuration if enabled and not already applied."""
        if not self.use_lora or isinstance(self.model, PeftModel):
            return

        lora_config = LoraConfig(
            r=self.lora_rank,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            task_type="CAUSAL_LM",
            inference_mode=False,
        )
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

    def forward(self, trajectories: List[Trajectory]) -> List[List[torch.Tensor]]:
        """
        Compute loss for each transition in the trajectories.
        
        Args:
            trajectories: List of trajectories containing states and actions
        
        Returns:
            List of lists of losses per trajectory transition
        """
        all_prompts = []
        all_actions = []
        trajectory_lengths = []

        # Prepare prompts and actions
        for traj in trajectories:
            trajectory_lengths.append(len(traj.states))
            for state, action in zip(traj.states, traj.actions):
                prompt = f"{state}<answer>"
                all_prompts.append(prompt)
                all_actions.append(action)

        # Tokenize prompts and full texts
        prompt_encodings = self.tokenizer(
            all_prompts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            add_special_tokens=True,
        )
        full_texts = [p + a for p, a in zip(all_prompts, all_actions)]
        full_encodings = self.tokenizer(
            full_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            add_special_tokens=True,
        )

        # Mask loss for prompt tokens
        labels = full_encodings.input_ids.clone()
        prompt_lengths = prompt_encodings.attention_mask.sum(dim=1)
        for i, prompt_len in enumerate(prompt_lengths):
            labels[i, :prompt_len] = -100

        # Move tensors to model's device
        input_ids = full_encodings.input_ids.to(self.model.device)
        attention_mask = full_encodings.attention_mask.to(self.model.device)
        labels = labels.to(self.model.device)

        # Compute loss
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        logits = outputs.logits
        shift_logits = logits[:, :-1].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        # Calculate per-token loss
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
        loss_per_token = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        ).view(shift_labels.shape)

        # Average loss per sequence
        non_ignored_mask = (shift_labels != -100).float()
        per_example_loss = (loss_per_token * non_ignored_mask).sum(dim=1) / non_ignored_mask.sum(dim=1).clamp(min=1e-6)

        # Group losses by trajectory
        losses = []
        start_idx = 0
        for length in trajectory_lengths:
            end_idx = start_idx + length
            losses.append(per_example_loss[start_idx:end_idx])
            start_idx = end_idx

        return losses

    def act(self, state: str, temperature: float = 1.0, top_p: float = 0.9, max_new_tokens: int = 100) -> str:
        """
        Generate action text given a state with randomness.
        
        Args:
            state: Current state description
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            max_new_tokens: Maximum tokens to generate
        
        Returns:
            Generated action text
        """
        prompt = f"{state}<answer>"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        # Configure generation
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            do_sample=True,
        )

        # Generate tokens
        outputs = self.model.generate(
            **inputs,
            generation_config=generation_config,
        )
        
        # Decode only the generated part
        generated_ids = outputs[:, inputs.input_ids.shape[1]:]
        return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    def save(self, path: str):
        """Save model and tokenizer to path."""
        os.makedirs(path, exist_ok=True)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(path)
        
        # Save model based on type
        if isinstance(self.model, PeftModel):
            self.model.save_pretrained(path)
            with open(os.path.join(path, "base_model_name.txt"), "w") as f:
                f.write(self.model_name)
        else:
            save_model(self.model, os.path.join(path, "model.safetensors"))

    def load(self, path: str):
        """Load model and tokenizer from path."""
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        
        # Check for PEFT model
        if os.path.exists(os.path.join(path, "adapter_config.json")):
            with open(os.path.join(path, "base_model_name.txt"), "r") as f:
                base_model_name = f.read().strip()
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                device_map="auto",
                trust_remote_code=True
            )
            self.model = PeftModel.from_pretrained(base_model, path)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                path,
                device_map="auto",
                trust_remote_code=True
            )
        self.model.eval()
'''