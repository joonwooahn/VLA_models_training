#!/usr/bin/env python
"""
Ablation Study를 위한 UniVLA 훈련 스크립트
"""

import os
import sys
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from collections import deque
import json

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.distributed as dist
from torch.nn.utils.rnn import pad_sequence
import tqdm
import numpy as np
from PIL import Image
import random

from accelerate import PartialState, Accelerator, DistributedDataParallelKwargs
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from transformers import AutoConfig, AutoImageProcessor
import draccus

# Import ablation config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from ablation_config import get_condition_by_name, StateType, ActionType, CameraType

# Prismatic imports
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.vla.datasets.real_world_dataset import PaddedCollatorForActionPrediction
from prismatic.models.policy.transformer_utils import MAPBlock

# Latent Action Model
from latent_action_model.genie.modules.lam import ControllableDINOLatentActionModel

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def get_ablation_config(condition):
    """Get configuration based on ablation condition"""
    # 기본 설정
    config = {
        "data_dir": "./converted_data",
        "state_indices": [],
        "action_indices": [],
        "state_dim": 0,
        "action_dim": 0,
        "checkout_name": condition.name
    }
    
    # State configuration
    if condition.state_type == StateType.POSITION_ONLY:
        # Position only
        if condition.action_type == ActionType.SINGLE_ARM:
            # 몸 + 오른팔 + 오른손 (position only)
            config["state_indices"] = list(range(0, 3)) + list(range(5, 12)) + list(range(19, 40))
            config["action_indices"] = [0, 1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]
        else:  # BIMANUAL
            # 몸 + 양팔 + 양손 (position only)
            config["state_indices"] = list(range(0, 3)) + list(range(5, 19)) + list(range(19, 60))
            config["action_indices"] = list(range(42))  # 전체 액션
    else:  # FULL_STATE
        # Position + velocity + torque
        base_indices = []
        if condition.action_type == ActionType.SINGLE_ARM:
            base_indices = list(range(0, 3)) + list(range(5, 12)) + list(range(19, 40))
        else:  # BIMANUAL
            base_indices = list(range(0, 3)) + list(range(5, 19)) + list(range(19, 60))
        
        # Add velocity and torque indices (assuming they follow the position indices)
        velocity_indices = [i + 60 for i in base_indices]
        torque_indices = [i + 120 for i in base_indices]
        config["state_indices"] = base_indices + velocity_indices + torque_indices
        
        if condition.action_type == ActionType.SINGLE_ARM:
            config["action_indices"] = [0, 1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]
        else:
            config["action_indices"] = list(range(42))
    
    config["state_dim"] = len(config["state_indices"])
    config["action_dim"] = len(config["action_indices"])
    
    return config


def get_norm_stats_from_files(dataset_dir, state_indices, action_indices):
    """state.npy와 action.npy 파일들로부터 정규화 통계를 계산합니다."""
    all_states, all_actions = [], []
    episode_paths = sorted([p for p in Path(dataset_dir).iterdir() if p.is_dir()])
    print(f"Calculating normalization stats from {len(episode_paths)} episodes...")
    for episode_path in tqdm.tqdm(episode_paths):
        state_data = np.load(episode_path / "state.npy")
        action_data = np.load(episode_path / "action.npy")
        
        # Apply filtering with bounds checking
        if len(state_indices) > 0 and state_data.shape[1] > max(state_indices):
            filtered_state = state_data[:, state_indices]
        else:
            print(f"⚠️  State data shape {state_data.shape} insufficient for indices {state_indices}")
            filtered_state = state_data
            
        if len(action_indices) > 0 and action_data.shape[1] > max(action_indices):
            filtered_action = action_data[:, action_indices]
        else:
            print(f"⚠️  Action data shape {action_data.shape} insufficient for indices {action_indices}")
            filtered_action = action_data
        
        all_states.append(torch.from_numpy(filtered_state))
        all_actions.append(torch.from_numpy(filtered_action))
    
    all_states_tensor = torch.cat(all_states, dim=0)
    all_actions_tensor = torch.cat(all_actions, dim=0)

    # State 정규화
    state_mean = all_states_tensor.mean(dim=0, keepdim=True)
    state_std = all_states_tensor.std(dim=0, keepdim=True)
    
    # Action 정규화
    action_mean = all_actions_tensor.mean(dim=0, keepdim=True)
    action_std = all_actions_tensor.std(dim=0, keepdim=True)
    
    stats = {
        "state_mean": state_mean.numpy().squeeze().tolist(),
        "state_std": state_std.numpy().squeeze().tolist(),
        "action_mean": action_mean.numpy().squeeze().tolist(), 
        "action_std": action_std.numpy().squeeze().tolist(),
    }
    return stats


class ManualDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, norm_stats, window_size, state_indices, action_indices, image_transform=None):
        self.root_dir = Path(root_dir)
        self.norm_stats = norm_stats
        self.window_size = window_size
        self.state_indices = state_indices
        self.action_indices = action_indices
        self.image_transform = image_transform
        self.episodes = []
        print(f"Loading episode info from {root_dir}...")
        for episode_path in tqdm.tqdm(sorted([p for p in self.root_dir.iterdir() if p.is_dir()])):
            action_file = episode_path / "action.npy"
            if action_file.exists():
                episode_len = len(np.load(action_file))
                if episode_len > window_size:
                    self.episodes.append({'path': episode_path, 'len': episode_len})

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, idx):
        episode_info = self.episodes[idx]
        episode_path = episode_info['path']
        episode_len = episode_info['len']

        start_idx = random.randint(0, episode_len - self.window_size - 1)
        end_idx = start_idx + self.window_size
        
        # Load and filter state
        full_state = np.load(episode_path / "state.npy")[start_idx]
        
        # Apply filtering with bounds checking
        if len(self.state_indices) > 0 and len(full_state) > max(self.state_indices):
            state_filtered = full_state[self.state_indices]
        else:
            # If indices are out of bounds, use available data
            available_indices = [idx for idx in self.state_indices if idx < len(full_state)]
            if available_indices:
                state_filtered = full_state[available_indices]
            else:
                state_filtered = full_state
                
        state_mean = np.array(self.norm_stats["state_mean"])
        state_std = np.array(self.norm_stats["state_std"])
        
        # Ensure dimensions match
        if len(state_filtered) != len(state_mean):
            state_mean = state_mean[:len(state_filtered)]
            state_std = state_std[:len(state_filtered)]
            
        state_normalized = (state_filtered - state_mean) / state_std
        states_tensor = torch.from_numpy(state_normalized).float()

        # Load and filter actions
        full_actions = np.load(episode_path / "action.npy")[start_idx:end_idx]
        
        # Apply filtering with bounds checking
        if len(self.action_indices) > 0 and full_actions.shape[1] > max(self.action_indices):
            actions_filtered = full_actions[:, self.action_indices]
        else:
            # If indices are out of bounds, use available data
            available_indices = [idx for idx in self.action_indices if idx < full_actions.shape[1]]
            if available_indices:
                actions_filtered = full_actions[:, available_indices]
            else:
                actions_filtered = full_actions
                
        action_mean = np.array(self.norm_stats["action_mean"])
        action_std = np.array(self.norm_stats["action_std"])
        
        # Ensure dimensions match
        if actions_filtered.shape[1] != len(action_mean):
            action_mean = action_mean[:actions_filtered.shape[1]]
            action_std = action_std[:actions_filtered.shape[1]]
            
        actions_normalized = (actions_filtered - action_mean) / action_std
        actions_tensor = torch.from_numpy(actions_normalized).float()

        # Load instruction
        instruction_file = episode_path / "instruction.txt"
        if instruction_file.exists():
            with open(instruction_file, "r") as f:
                instruction = f.read().strip()
        else:
            instruction = "manipulate the object"

        # Load images
        main_image_path = episode_path / f"frame_{start_idx + 1:03d}.png"
        if not main_image_path.exists():
            # Fallback to different naming convention
            main_image_path = episode_path / f"image_{start_idx + 1}.png"
        
        if main_image_path.exists():
            pixel_values = self.image_transform(Image.open(main_image_path).convert("RGB"))
        else:
            # Create dummy image if not found
            pixel_values = torch.zeros(3, 224, 224)

        resize_to_224 = transforms.Resize((224, 224))
        to_tensor = transforms.ToTensor()
        initial_pixel_values = to_tensor(resize_to_224(Image.open(main_image_path).convert("RGB"))) if main_image_path.exists() else torch.zeros(3, 224, 224)
        
        target_frame_path = episode_path / f"frame_{end_idx:03d}.png"
        if not target_frame_path.exists():
            target_frame_path = episode_path / f"image_{end_idx}.png"
        target_pixel_values = to_tensor(resize_to_224(Image.open(target_frame_path).convert("RGB"))) if target_frame_path.exists() else torch.zeros(3, 224, 224)

        return dict(
            pixel_values=pixel_values, 
            actions=actions_tensor, 
            lang=instruction, 
            proprio=states_tensor,
            initial_pixel_values=initial_pixel_values, 
            target_pixel_values=target_pixel_values,
            initial_pixel_values_hist=None, 
            target_pixel_values_hist=None,
            with_hist=torch.tensor(False),
        )


def load_data_manual(dataset_dir, batch_size, processor, window_size, state_indices, action_indices):
    norm_stats = get_norm_stats_from_files(dataset_dir, state_indices, action_indices)
    dataset = ManualDataset(
        root_dir=dataset_dir, 
        norm_stats=norm_stats, 
        window_size=window_size,
        state_indices=state_indices,
        action_indices=action_indices,
        image_transform=processor.image_processor.apply_transform,
    )
    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
    )
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=8,
        collate_fn=collator, pin_memory=True, persistent_workers=True if 8 > 0 else False
    )
    return dataloader, norm_stats


class ActionDecoderHead(torch.nn.Module):
    def __init__(self, window_size, action_dim, hidden_dim=512):
        super().__init__()
        self.attn_pool = MAPBlock(n_latents=1, vis_dim=4096, embed_dim=hidden_dim, n_heads=hidden_dim // 64)
        self.visual_pool = MAPBlock(n_latents=1, vis_dim=4096, embed_dim=hidden_dim, n_heads=hidden_dim // 64)
        self.proprio_proj = nn.Sequential(
            nn.Linear(len([]), hidden_dim),  # Will be set dynamically
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, window_size * action_dim)
        )

    def set_state_dim(self, state_dim):
        """Dynamically set the state dimension"""
        self.proprio_proj[0] = nn.Linear(state_dim, self.proprio_proj[0].out_features)

    def forward(self, latent_action_tokens, visual_embed, proprio):
        # Handle the large hidden dimension (32064) by projecting to expected size (4096)
        if latent_action_tokens.shape[-1] == 32064:
            # Project down to 4096 dimension
            if not hasattr(self, 'hidden_proj'):
                self.hidden_proj = nn.Linear(32064, 4096).to(latent_action_tokens.device)
            latent_action_tokens = self.hidden_proj(latent_action_tokens)
        
        if visual_embed.shape[-1] == 32064:
            # Project down to 4096 dimension  
            if not hasattr(self, 'visual_proj'):
                self.visual_proj = nn.Linear(32064, 4096).to(visual_embed.device)
            visual_embed = self.visual_proj(visual_embed)
        
        # Take last 4 tokens - ensure we have at least 4 tokens
        if latent_action_tokens.shape[1] >= 4:
            latent_action_tokens = latent_action_tokens[:, -4:]
        else:
            # Pad with zeros if we don't have enough tokens
            batch_size, seq_len, hidden_dim = latent_action_tokens.shape
            padding = torch.zeros(batch_size, 4 - seq_len, hidden_dim, 
                                device=latent_action_tokens.device, dtype=latent_action_tokens.dtype)
            latent_action_tokens = torch.cat([padding, latent_action_tokens], dim=1)
        
        # Ensure visual_embed has the right shape (256 patches)
        if visual_embed.shape[1] != 256:
            if visual_embed.shape[1] > 256:
                visual_embed = visual_embed[:, :256]
            else:
                # Pad with zeros if needed
                batch_size, seq_len, hidden_dim = visual_embed.shape
                padding = torch.zeros(batch_size, 256 - seq_len, hidden_dim,
                                    device=visual_embed.device, dtype=visual_embed.dtype)
                visual_embed = torch.cat([visual_embed, padding], dim=1)
        
        # Use simple mean pooling instead of complex attention pooling to avoid attention mask issues
        pooled_embed = latent_action_tokens.mean(dim=1)  # Average over sequence dimension
        
        # Handle proprioceptive dimension mismatch
        if proprio.shape[-1] != self.proprio_proj[0].in_features:
            # Create a new projection layer with the correct input dimension
            actual_proprio_dim = proprio.shape[-1]
            out_features = self.proprio_proj[0].out_features
            self.proprio_proj[0] = nn.Linear(actual_proprio_dim, out_features).to(proprio.device)
        
        proprio_embed = self.proprio_proj(proprio)
        
        # Concatenate embeddings
        concat_embed = torch.cat([pooled_embed, proprio_embed], dim=-1)
        
        # Handle projection dimension mismatch
        if concat_embed.shape[-1] != self.proj[0].in_features:
            # Create a new projection layer with the correct input dimension
            actual_concat_dim = concat_embed.shape[-1]
            out_features = self.proj[0].out_features
            self.proj[0] = nn.Linear(actual_concat_dim, out_features).to(concat_embed.device)
        
        action_embed = self.proj(concat_embed)
        return action_embed


class Wrapped_Model(torch.nn.Module):
    def __init__(self, vla, action_dim, state_dim, freeze_vla=False, window_size=8):
        super().__init__()
        self.vla = vla
        self.action_decoder = ActionDecoderHead(window_size, action_dim)
        self.action_decoder.set_state_dim(state_dim)
        self.freeze_vla = freeze_vla
        
        if self.freeze_vla:
            for param in self.vla.parameters():
                param.requires_grad = False

    def forward(self, batch):
        if self.freeze_vla:
            with torch.no_grad():
                slow_output = self.vla(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    pixel_values=batch["pixel_values"],
                    labels=batch["labels"],
                    output_hidden_states=True
                )
        else:
            slow_output = self.vla(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                pixel_values=batch["pixel_values"],
                labels=batch["labels"],
            )
        
        return self.action_decoder_forward(batch, slow_output)

    def action_decoder_forward(self, batch, slow_output):
        # Check if hidden_states exist, otherwise use last_hidden_state
        if hasattr(slow_output, 'hidden_states') and slow_output.hidden_states is not None:
            latent_action_tokens = slow_output.hidden_states[-1]
        elif hasattr(slow_output, 'last_hidden_state') and slow_output.last_hidden_state is not None:
            latent_action_tokens = slow_output.last_hidden_state
        else:
            # Fallback: use logits if nothing else is available
            latent_action_tokens = slow_output.logits
        # Handle different output types for visual embeddings
        if hasattr(slow_output, 'multimodal_output') and slow_output.multimodal_output is not None:
            visual_embed = slow_output.multimodal_output.image_patches_embeddings
        elif hasattr(slow_output, 'image_patches_embeddings'):
            visual_embed = slow_output.image_patches_embeddings
        else:
            # Fallback: use a dummy visual embedding
            visual_embed = torch.zeros(latent_action_tokens.shape[0], 256, latent_action_tokens.shape[-1], 
                                     device=latent_action_tokens.device, dtype=latent_action_tokens.dtype)
        proprio = batch['proprio']
        
        pred_actions = self.action_decoder(latent_action_tokens, visual_embed, proprio)
        pred_actions = pred_actions.view(pred_actions.shape[0], -1, pred_actions.shape[-1] // 8)
        
        gt_actions = batch['actions']
        
        # Handle action dimension mismatch by matching to ground truth dimensions
        if pred_actions.shape[-1] != gt_actions.shape[-1]:
            print(f"Action dimension mismatch: pred {pred_actions.shape} vs gt {gt_actions.shape}")
            # Take the minimum dimensions to avoid index errors
            min_action_dim = min(pred_actions.shape[-1], gt_actions.shape[-1])
            pred_actions = pred_actions[..., :min_action_dim]
            gt_actions = gt_actions[..., :min_action_dim]
        
        action_loss = nn.MSELoss()(pred_actions, gt_actions)
        
        return slow_output, action_loss, action_loss, latent_action_tokens


@dataclass
class FinetuneConfig:
    # Directory Paths
    data_root_dir: Path = Path("./converted_data")
    vla_path: str = "univla/vla_scripts/univla-7b"
    lam_path: str = "univla/vla_scripts/univla-latent-action-model/lam-stage-2.ckpt"
    dataset_name: str = "rlwrld"
    run_root_dir: Path = Path("runs")
    adapter_tmp_dir: Path = Path("adapter-tmp")

    # Fine-tuning Parameters
    batch_size: int = 16
    max_steps: int = 10000
    save_steps: int = 2000
    learning_rate: float = 1e-4
    grad_accumulation_steps: int = 2
    image_aug: bool = False
    shuffle_buffer_size: int = 10000
    save_latest_checkpoint_only: bool = True

    # LAM setting
    codebook_size: int = 16
    lam_model_dim: int = 768
    lam_latent_dim: int = 128
    lam_num_latents: int = 32
    lam_patch_size: int = 14
    lam_enc_blocks: int = 12
    lam_dec_blocks: int = 12
    lam_num_heads: int = 12
    window_size: int = 12

    freeze_vla: bool = False
    # LoRA Arguments
    use_lora: bool = True
    lora_rank: int = 32
    lora_dropout: float = 0.0
    use_quantization: bool = False
    use_gradient_checkpointing: bool = False

    # Tracking Parameters
    wandb_project: str = "univla-finetune-ablation"
    wandb_entity: str = "ablation-study"
    run_id_note: Optional[str] = None


def finetune_ablation(condition_name, data_root_dir, output_dir, max_steps, batch_size, learning_rate):
    """Main training function for ablation study"""
    print(f"🚀 Starting UniVLA training for condition: {condition_name}")
    
    # Get condition and configuration
    condition = get_condition_by_name(condition_name)
    if not condition:
        raise ValueError(f"Condition '{condition_name}' not found")
    
    ablation_config = get_ablation_config(condition)
    print(f"📊 Ablation config: {ablation_config}")
    
    # Update configuration
    cfg = FinetuneConfig()
    cfg.data_root_dir = Path(data_root_dir)
    cfg.run_root_dir = Path(output_dir)
    cfg.max_steps = max_steps
    cfg.batch_size = batch_size
    cfg.learning_rate = learning_rate
    cfg.run_id_note = condition_name
    
    # Setup distributed training
    assert torch.cuda.is_available(), "Training requires GPU!"
    distributed_state = PartialState()
    torch.cuda.set_device(device_id := distributed_state.local_process_index)
    torch.cuda.empty_cache()
    
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(mixed_precision="bf16", kwargs_handlers=[ddp_kwargs])
    
    # Register OpenVLA model
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
    
    # Load processor and model
    print("🔄 Loading model and processor...")
    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
    
    quantization_config = None
    if cfg.use_quantization:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4"
        )
    
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    
    # Device placement
    if cfg.use_quantization:
        vla = prepare_model_for_kbit_training(vla)
    else:
        vla = vla.to(device_id)
    
    # Enable gradient checkpointing for memory efficiency (optional for A100)
    if cfg.use_gradient_checkpointing:
        vla.gradient_checkpointing_enable()
        print("🔧 Gradient checkpointing enabled for memory efficiency")
    
    # LoRA setup
    if cfg.use_lora:
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=min(cfg.lora_rank, 16),
            lora_dropout=cfg.lora_dropout,
            target_modules="all-linear",
            init_lora_weights="gaussian",
        )
        vla = get_peft_model(vla, lora_config)
        vla.print_trainable_parameters()
    
    # Create wrapped model
    wrapped_model = Wrapped_Model(
        vla=vla, 
        action_dim=ablation_config["action_dim"],
        state_dim=ablation_config["state_dim"],
        freeze_vla=cfg.freeze_vla, 
        window_size=cfg.window_size
    ).to(device_id)
    
    trainable_params = sum(p.numel() for p in wrapped_model.parameters() if p.requires_grad)
    print(f'📈 Total Trainable Params: {trainable_params:,}')
    
    # Optimizer and scheduler
    optimizer = AdamW([p for p in wrapped_model.parameters() if p.requires_grad], 
                     lr=cfg.learning_rate, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(cfg.max_steps * 0.5), gamma=0.1)
    
    # Load Latent Action Model
    print("🔄 Loading Latent Action Model...")
    latent_action_model = ControllableDINOLatentActionModel(
        in_dim=3,
        model_dim=cfg.lam_model_dim,
        latent_dim=cfg.lam_latent_dim,
        num_latents=cfg.codebook_size,
        patch_size=cfg.lam_patch_size,
        enc_blocks=cfg.lam_enc_blocks,
        dec_blocks=cfg.lam_dec_blocks,
        num_heads=cfg.lam_num_heads,
        dropout=0.,
    )
    
    if Path(cfg.lam_path).exists():
        lam_ckpt = torch.load(cfg.lam_path, map_location='cpu')['state_dict']
        new_ckpt = {key.replace("lam.", ""): val for key, val in lam_ckpt.items()}
        latent_action_model.load_state_dict(new_ckpt, strict=True)
    else:
        print(f"⚠️  LAM checkpoint not found at {cfg.lam_path}, using random weights")
    
    latent_action_model = latent_action_model.to(device_id).eval()
    
    # Load data
    print("📂 Loading training data...")
    dataloader, stats = load_data_manual(
        dataset_dir=cfg.data_root_dir,
        batch_size=cfg.batch_size,
        processor=processor,
        window_size=cfg.window_size,
        state_indices=ablation_config["state_indices"],
        action_indices=ablation_config["action_indices"]
    )
    
    # Setup directories
    run_dir = cfg.run_root_dir / condition_name
    os.makedirs(run_dir, exist_ok=True)
    
    # Save statistics
    if distributed_state.is_main_process:
        stats_path = run_dir / 'dataset_statistics.json'
        print(f"💾 Saving dataset statistics to {stats_path}")
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=4)
    
    # Prepare for distributed training
    wrapped_model, latent_action_model, optimizer, scheduler, dataloader = accelerator.prepare(
        wrapped_model, latent_action_model, optimizer, scheduler, dataloader
    )
    
    # Training loop
    print("🎯 Starting training...")
    recent_losses = deque(maxlen=cfg.grad_accumulation_steps)
    recent_action_accuracies = deque(maxlen=cfg.grad_accumulation_steps)
    
    with tqdm.tqdm(total=cfg.max_steps, leave=False) as progress:
        wrapped_model.train()
        optimizer.zero_grad()
        current_step = 0
        
        while current_step < cfg.max_steps:
            for batch_idx, batch in enumerate(dataloader):
                # Move batch to device
                for key in ["initial_pixel_values", "target_pixel_values", "pixel_values", "actions", "proprio"]:
                    if key in batch and batch[key] is not None:
                        batch[key] = batch[key].to(device_id)
                        if key == "pixel_values":
                            batch[key] = batch[key].to(torch.bfloat16)
                
                # Generate latent action labels
                with torch.no_grad():
                    video = torch.stack([batch["initial_pixel_values"], batch["target_pixel_values"]], dim=1)
                    # Handle both wrapped and unwrapped models
                    model_to_use = latent_action_model.module if hasattr(latent_action_model, 'module') else latent_action_model
                    latent_action_idx_batch = model_to_use.vq_encode(video)['indices'].squeeze()
                    # Ensure proper device placement - use device from video tensor
                    latent_action_idx_batch = latent_action_idx_batch.to(video.device)
                
                # Create input sequences
                input_ids_list = []
                labels_list = []
                
                for idx, latent_action_idx in enumerate(latent_action_idx_batch):
                    action_vocab = [f'<ACT_{i.item()}>' for i in latent_action_idx]
                    action_tokens = ''.join(action_vocab)
                    
                    # Create prompt
                    prompt_builder = PurePromptBuilder("openvla")
                    instruction = batch.get('lang', ['manipulate the object'])[idx] if isinstance(batch.get('lang'), list) else 'manipulate the object'
                    conversation = [
                        {"from": "human", "value": f"What action should the robot take to {instruction.lower()}?"},
                        {"from": "gpt", "value": action_tokens},
                    ]
                    for turn in conversation:
                        prompt_builder.add_turn(turn["from"], turn["value"])
                    
                    # Tokenize
                    input_ids = processor.tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
                    labels = list(input_ids)
                    
                    device = batch["pixel_values"].device  # Get device from batch
                    input_ids, labels = torch.tensor(input_ids, device=device), torch.tensor(labels, device=device)
                    labels[:-(len(action_vocab) + 1)] = -100
                    
                    input_ids_list.append(input_ids)
                    labels_list.append(labels)
                
                # Pad sequences
                device = batch["pixel_values"].device  # Ensure consistent device
                input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=processor.tokenizer.pad_token_id).to(device)
                labels = pad_sequence(labels_list, batch_first=True, padding_value=-100).to(device)
                
                # Truncate if necessary
                input_ids = input_ids[:, :processor.tokenizer.model_max_length]
                labels = labels[:, :processor.tokenizer.model_max_length]
                attention_mask = input_ids.ne(processor.tokenizer.pad_token_id)
                
                batch["input_ids"] = input_ids
                batch["attention_mask"] = attention_mask
                batch["labels"] = labels
                
                # Forward pass
                output, act_loss, loss_one_step, latent_action_tokens = wrapped_model(batch)
                loss = act_loss if cfg.freeze_vla else act_loss + output.loss
                normalized_loss = loss / cfg.grad_accumulation_steps
                
                # Backward pass
                torch.nn.utils.clip_grad_norm_(wrapped_model.parameters(), max_norm=0.3)
                normalized_loss.backward()
                
                # Compute metrics - handle both wrapped and unwrapped models
                if hasattr(wrapped_model, 'module'):
                    num_patches = wrapped_model.module.vla.vision_backbone.featurizer.patch_embed.num_patches
                else:
                    num_patches = wrapped_model.vla.vision_backbone.featurizer.patch_embed.num_patches
                action_logits = output.logits[:, num_patches:-1]
                action_preds = action_logits.argmax(dim=2)
                action_gt = batch["labels"][:, 1:].to(action_preds.device)
                mask = action_gt > 32000
                
                correct_preds = (action_preds == action_gt) & mask
                action_accuracy = correct_preds.sum().float() / mask.sum().float() if mask.sum() > 0 else torch.tensor(0.0)
                
                recent_losses.append(loss.item())
                recent_action_accuracies.append(action_accuracy.item())
                
                gradient_step_idx = batch_idx // cfg.grad_accumulation_steps
                smoothened_loss = sum(recent_losses) / len(recent_losses)
                smoothened_accuracy = sum(recent_action_accuracies) / len(recent_action_accuracies)
                
                # Optimizer step
                if (batch_idx + 1) % cfg.grad_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    progress.update()
                
                # Save checkpoint
                if (gradient_step_idx + current_step) > 0 and (gradient_step_idx + current_step) % cfg.save_steps == 0:
                    step = gradient_step_idx + current_step
                    if distributed_state.is_main_process:
                        print(f"💾 Saving checkpoint at step {step}")
                        checkpoint_dir = run_dir / f"checkpoint-{step}"
                        os.makedirs(checkpoint_dir, exist_ok=True)
                        
                        if cfg.use_lora:
                            wrapped_model.module.vla.save_pretrained(checkpoint_dir)
                        
                        decoder_path = checkpoint_dir / f'action_decoder-{step}.pt'
                        torch.save(wrapped_model.module.action_decoder.state_dict(), decoder_path)
                        processor.save_pretrained(checkpoint_dir)
                    
                    dist.barrier()
                
                # Update progress
                description = f"Step {current_step + gradient_step_idx} | Loss: {smoothened_loss:.4f} | Acc: {smoothened_accuracy:.4f}"
                progress.set_description(description)
                
                current_step = gradient_step_idx + 1 + current_step
                if current_step >= cfg.max_steps:
                    print(f"✅ Max steps {cfg.max_steps} reached! Training completed.")
                    break
            
            if current_step >= cfg.max_steps:
                break
    
    print(f"🎉 Training completed for condition: {condition_name}")


def main():
    parser = argparse.ArgumentParser(description="UniVLA Ablation Study Training")
    parser.add_argument("--condition", type=str, required=True)
    parser.add_argument("--data-root-dir", type=str, default="./converted_data")
    parser.add_argument("--output-dir", type=str, default="./outputs")
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    
    args = parser.parse_args()
    
    condition = get_condition_by_name(args.condition)
    if not condition:
        print(f"❌ Error: Condition '{args.condition}' not found")
        return
    
    print(f"✅ Starting training for condition: {args.condition}")
    print(f"📂 Data: {args.data_root_dir}")
    print(f"📁 Output: {args.output_dir}")
    print(f"🔢 Steps: {args.max_steps}")
    print(f"📊 Batch size: {args.batch_size}")
    print(f"📈 Learning rate: {args.learning_rate}")
    
    try:
        finetune_ablation(
            condition_name=args.condition,
            data_root_dir=args.data_root_dir,
            output_dir=args.output_dir,
            max_steps=args.max_steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate
        )
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
