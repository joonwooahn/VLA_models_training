import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import draccus
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.distributed as dist
from torch.nn.utils.rnn import pad_sequence
import tqdm
import pickle
from ema_pytorch import EMA
from accelerate import PartialState, Accelerator
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from transformers import AutoConfig, AutoImageProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast

import wandb
from prismatic.models.backbones.llm.prompting import PurePromptBuilder, VicunaV15ChatPromptBuilder
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics
# from prismatic.vla.datasets.real_world_dataset import find_all_hdf5, load_data_univla


from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from prismatic.models.policy.transformer_utils import MAPBlock
import numpy as np

# === 새로운 데이터 처리 코드 시작 ===
import glob
import numpy as np
from PIL import Image
from prismatic.vla.datasets.real_world_dataset import PaddedCollatorForActionPrediction # 원본의 collate 함수를 그대로 사용
import random
import json

################################################################################
### allex
# DATA_DIR = "/virtual_lab/rlwrld/david/pi_0_fast/openpi/data/rlwrld_dataset/allex-cube-dataset_multiview_converted_state_action"
# CHECKOUT_NAME = "allex_state_action_filter"

# DATA_DIR = "/virtual_lab/rlwrld/david/pi_0_fast/openpi/data/rlwrld_dataset/allex-cube-dataset_single_view_converted_state_action"
# CHECKOUT_NAME = "allex_state_action_filter_single_view"

# DATA_DIR = "/virtual_lab/rlwrld/david/pi_0_fast/openpi/data/rlwrld_dataset/allex-cube-dataset_side_view_converted_state_action"
# CHECKOUT_NAME = "allex_state_action_filter_side_view"

### 데이터에서 오른팔 및 손 움직임만 훈련하기 위해 인덱스 설정
# INDICES_FOR_STATE = list(range(4)) + list(range(6, 13)) + list(range(20, 40)) 
# INDICES_FOR_ACTION = [0, 1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]
################################################################################

################################################################################
### gr1
## DATA_DIR = "/virtual_lab/rlwrld/david/pi_0_fast/openpi/data/rlwrld_dataset/gr1-cube-dataset_multiview_converted_state_action"
# CHECKOUT_NAME = "gr1_state_action_filter"

# DATA_DIR = "/virtual_lab/rlwrld/david/pi_0_fast/openpi/data/rlwrld_dataset/gr1-cube-dataset_single_view_converted_state_action"
# CHECKOUT_NAME = "gr1_state_action_filter_single_view"

DATA_DIR = "/virtual_lab/rlwrld/david/pi_0_fast/openpi/data/rlwrld_dataset/gr1-cube-dataset_side_view_converted_state_action"
CHECKOUT_NAME = "gr1_state_action_filter_side_view"

# ### 데이터에서 오른팔 및 손 움직임만 훈련하기 위해 인덱스 설정
INDICES_FOR_STATE = list(range(13)) + list(range(20, 31))
INDICES_FOR_ACTION = [0, 1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17]
################################################################################

STATE_DIM = len(INDICES_FOR_STATE)
ACTION_DIM = len(INDICES_FOR_ACTION)

# ==============================================================================
# 커스텀 데이터 처리 함수 및 클래스
# ==============================================================================
def get_norm_stats_from_files(dataset_dir):
    """ state.npy와 action.npy 파일들로부터 정규화 통계를 계산합니다. """
    all_states, all_actions = [], []
    episode_paths = sorted([p for p in Path(dataset_dir).iterdir() if p.is_dir()])
    print(f"Calculating normalization stats from {len(episode_paths)} episodes...")
    for episode_path in tqdm.tqdm(episode_paths):
        all_states.append(torch.from_numpy(np.load(episode_path / "state.npy")))
        all_actions.append(torch.from_numpy(np.load(episode_path / "action.npy")))
    
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
    def __init__(self, root_dir, norm_stats, window_size, image_transform=None):
        self.root_dir = Path(root_dir)
        self.norm_stats = norm_stats
        self.window_size = window_size
        self.image_transform = image_transform
        self.episodes = []
        print(f"Loading episode info from {root_dir}...")
        for episode_path in tqdm.tqdm(sorted([p for p in self.root_dir.iterdir() if p.is_dir()])):
            episode_len = len(np.load(episode_path / "action.npy"))
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
        
        full_state = np.load(episode_path / "state.npy")[start_idx]
        state_filtered = full_state[INDICES_FOR_STATE]
        state_mean = np.array(self.norm_stats["state_mean"])[INDICES_FOR_STATE]
        state_std = np.array(self.norm_stats["state_std"])[INDICES_FOR_STATE]
        state_normalized = (state_filtered - state_mean) / state_std
        states_tensor = torch.from_numpy(state_normalized).float()

        full_actions = np.load(episode_path / "action.npy")[start_idx:end_idx]
        actions_filtered = full_actions[:, INDICES_FOR_ACTION]
        action_mean = np.array(self.norm_stats["action_mean"])[INDICES_FOR_ACTION]
        action_std = np.array(self.norm_stats["action_std"])[INDICES_FOR_ACTION]
        actions_normalized = (actions_filtered - action_mean) / action_std
        actions_tensor = torch.from_numpy(actions_normalized).float()

        with open(episode_path / "instruction.txt", "r") as f:
            instruction = f.read().strip()

        main_image_path = episode_path / f"frame_{start_idx + 1:03d}.png"
        pixel_values = self.image_transform(Image.open(main_image_path).convert("RGB"))

        resize_to_224 = transforms.Resize((224, 224))
        to_tensor = transforms.ToTensor()
        initial_pixel_values = to_tensor(resize_to_224(Image.open(main_image_path).convert("RGB")))
        target_frame_path = episode_path / f"frame_{end_idx:03d}.png"
        target_pixel_values = to_tensor(resize_to_224(Image.open(target_frame_path).convert("RGB")))

        return dict(
            pixel_values=pixel_values, actions=actions_tensor, lang=instruction, proprio=states_tensor,
            initial_pixel_values=initial_pixel_values, target_pixel_values=target_pixel_values,
            initial_pixel_values_hist=None, target_pixel_values_hist=None,
            with_hist=torch.tensor(False),
        )

def load_data_manual(dataset_dir, batch_size, processor, window_size):
    norm_stats = get_norm_stats_from_files(dataset_dir)
    dataset = ManualDataset(
        root_dir=dataset_dir, norm_stats=norm_stats, window_size=window_size,
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
    def __init__(self, window_size, hidden_dim=512):
        super().__init__()
        self.attn_pool = MAPBlock(n_latents=1, vis_dim=4096, embed_dim=hidden_dim, n_heads=hidden_dim // 64)
        self.visual_pool = MAPBlock(n_latents=1, vis_dim=4096, embed_dim=hidden_dim, n_heads=hidden_dim // 64)
        self.proprio_proj = nn.Sequential(
            nn.Linear(STATE_DIM, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, window_size * ACTION_DIM)
        )

    def forward(self, latent_action_tokens, visual_embed, proprio):
        # 수정: 추론 코드와 동일하게 마지막 4개 토큰만 사용하도록 로직 추가
        latent_action_tokens = latent_action_tokens[:, -4:]
        proprio = self.proprio_proj(proprio)
        visual_embed = self.visual_pool(visual_embed)
        action = self.proj(torch.cat([self.attn_pool(latent_action_tokens, init_embed=visual_embed), proprio], dim=-1))
        return action


class Wrapped_Model(torch.nn.Module):
    def __init__(self, vla, freeze_vla=False, window_size=12):
        super().__init__()
        self.vla = vla
        self.window_size = window_size
        # 수정: ActionDecoderHead를 사용하도록 변경
        self.action_decoder = ActionDecoderHead(window_size=window_size)
        if freeze_vla:
            self.vla.requires_grad_(False)

    def forward(self, batch):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            vla_output = self.vla(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                pixel_values=batch["pixel_values"],
                labels=batch["labels"],
                output_hidden_states = True,        # Return intermediate tokens of all layers
            )
        loss, loss_one_step, latent_action_tokens = self.action_decoder_forward(batch, vla_output)

        return vla_output, loss, loss_one_step, latent_action_tokens

    def action_decoder_forward(self, batch, slow_output):
        # Task and action latents
        visual_embed = slow_output.hidden_states[-1][:, : self.vla.vision_backbone.featurizer.patch_embed.num_patches ].to(torch.float)
        latent_tokens = slow_output.hidden_states[-1][:, self.vla.vision_backbone.featurizer.patch_embed.num_patches : ]
        action_gt = batch["labels"].to(latent_tokens.device)
        mask = action_gt > 32000

        latent_action_tokens = []
        for idx, per_sample_latent_tokens in enumerate(latent_tokens):
            per_sample_latent_action_tokens = per_sample_latent_tokens[mask[idx], :]
            latent_action_tokens.append(per_sample_latent_action_tokens)
        latent_action_tokens = torch.stack(latent_action_tokens).to(torch.float)

        # pred_action = self.action_decoder(latent_action_tokens, visual_embed).reshape(-1, self.window_size, 7)
        # action_decoder 호출 시 proprio 제거, reshape 차원을 ACTION_DIM로 변경
        # action_decoder_forward 함수 내부
        pred_action = self.action_decoder(
            latent_action_tokens, 
            visual_embed, 
            batch['proprio'] # <--- proprio 전달 다시 추가
        ).reshape(-1, self.window_size, ACTION_DIM) # <--- 차원을 ACTION_DIM (21)으로

        loss = torch.nn.functional.l1_loss(pred_action, batch['actions'], reduction='none')
        loss_one_step = loss[:,0].mean()
        loss = loss.mean()

        return loss, loss_one_step, latent_action_tokens


@dataclass
class FinetuneConfig:
    # Directory Paths
    # data_root_dir: Path = Path("/path/to/your/local/hdf5_data")     # Path to Open-X dataset directory
    data_root_dir: Path = DATA_DIR     # Path to Open-X dataset directory

    # vla_path: str = "/path/to/your/pretrained-univla-7b"            # Path to your local UniVLA path
    vla_path: str = "./vla_scripts/univla-7b"            # Path to your local UniVLA path
    
    # lam_path: str = "latent_action_model/logs/task_centric_lam_stage2/epoch=0-step=200000.ckpt"
    lam_path: str = "./vla_scripts/univla-latent-action-model/lam-stage-2.ckpt"
    dataset_name: str = "rlwrld"                                    # Name of fine-tuning dataset (e.g., `droid_wipe`)
    run_root_dir: Path = Path("runs")                               # Path to directory to store logs & checkpoints
    adapter_tmp_dir: Path = Path("adapter-tmp")                     # Temporary directory for LoRA weights before fusing

    # Fine-tuning Parameters
    batch_size: int = 16                                             # Fine-tuning batch size
    max_steps: int = 30001                                          # Max number of fine-tuning steps
    save_steps: int = 5000                                          # Interval for checkpoint saving
    learning_rate: float = 3.5e-4                                   # Fine-tuning learning rate
    grad_accumulation_steps: int = 2                                # Gradient accumulation steps
    image_aug: bool = False                                         # Whether to train with image augmentations
    shuffle_buffer_size: int = 100_00                               # Dataloader shuffle buffer size (can reduce if OOM)
    save_latest_checkpoint_only: bool = True                        # Whether to save only one checkpoint per run and
                                                                    #   continually overwrite the latest checkpoint
                                                                    #   (If False, saves all checkpoints)
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
    use_lora: bool = True                                           # Whether to use LoRA fine-tuning
    lora_rank: int = 32                                             # Rank of LoRA weight matrix
    lora_dropout: float = 0.0                                       # Dropout applied to LoRA weights
    use_quantization: bool = False                                  # Whether to 4-bit quantize VLA for LoRA fine-tuning
                                                                    #   => CAUTION: Reduces memory but hurts performance

    # hdf5 data config                                                             
    # camera_names: str = "camera_high"

    # Tracking Parameters
    wandb_project: str = "univla-finetune"                          # Name of W&B project to log to (use default!)
    wandb_entity: str = "joonwoo-ahn"                              # Name of entity to log under
    run_id_note: Optional[str] = CHECKOUT_NAME                               # Extra note for logging, Weights & Biases
    # run_id_note: Optional[str] = "norm_stats"



@draccus.wrap()
def finetune(cfg: FinetuneConfig) -> None:
    print(f"Fine-tuning UniVLA Model `{cfg.vla_path}` on `{cfg.dataset_name}`")

    # [Validate] Ensure GPU Available & Set Device / Distributed Context
    assert torch.cuda.is_available(), "Fine-tuning assumes at least one GPU is available!"
    distributed_state = PartialState()

    if distributed_state.is_main_process:
        print("This is the main process (rank 0).")
    else:
        print(f"This is a worker process (rank {distributed_state.process_index}).")
    
    torch.cuda.set_device(device_id := distributed_state.local_process_index)
    torch.cuda.empty_cache()

    from accelerate import DistributedDataParallelKwargs

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(mixed_precision="bf16", kwargs_handlers=[ddp_kwargs])

    # Configure Unique Experiment ID & Log Directory
    exp_id = f"{cfg.run_id_note}"
    # exp_id = (
    #     # f"{cfg.vla_path.split('/')[-1]}+{cfg.dataset_name}"
    #     # f"+b{cfg.batch_size * cfg.grad_accumulation_steps}"
    #     # f"+lr-{cfg.learning_rate}"
    # )
    if cfg.use_lora:
        exp_id += f"+lora-r{cfg.lora_rank}+dropout-{cfg.lora_dropout}"
    if cfg.use_quantization:
        exp_id += "+q-4bit"
    if cfg.run_id_note is not None:
        exp_id += f"--{cfg.run_id_note}"
    if cfg.image_aug:
        exp_id += "--image_aug"

    # exp_id += f'=w-LowLevelDecoder-ws-{cfg.window_size}'

    # Start =>> Build Directories
    run_dir, adapter_dir = cfg.run_root_dir / exp_id, cfg.adapter_tmp_dir / exp_id
    os.makedirs(run_dir, exist_ok=True)

    # Quantization Config =>> only if LoRA fine-tuning
    quantization_config = None
    if cfg.use_quantization:
        assert cfg.use_lora, "Quantized training only supported for LoRA fine-tuning!"
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4"
        )

    # Register OpenVLA model to HF Auto Classes (not needed if the model is on HF Hub)
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    # Load OpenVLA Processor and Model using HF AutoClasses
    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )


    # Device Placement =>> note that BitsAndBytes automatically handles for quantized training
    if cfg.use_quantization:
        vla = prepare_model_for_kbit_training(vla)
    else:
        vla = vla.to(device_id)

    # [LoRA] Wrap Model w/ PEFT `LoraConfig` =>> by default we set `target_modules=all-linear`
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

    # Create Action Tokenizer
    action_tokenizer = ActionTokenizer(processor.tokenizer)
    wrapped_model = Wrapped_Model(vla = vla, freeze_vla = cfg.freeze_vla, window_size=cfg.window_size).to(device_id)
    
    trainable_total_params = sum(p.numel() for p in wrapped_model.parameters() if p.requires_grad)
    print('Total Trainable Params: ', trainable_total_params)
    
    # Create Optimizer =>> note that we default to a simple constant learning rate!
    trainable_params = [param for param in wrapped_model.parameters() if param.requires_grad]
    optimizer = AdamW(trainable_params, lr=cfg.learning_rate, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = int(cfg.max_steps * 8 * 0.5), gamma=0.1)
        
    from latent_action_model.genie.modules.lam import ControllableDINOLatentActionModel

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

    lam_ckpt = torch.load(cfg.lam_path)['state_dict']
    new_ckpt = {}
    for key in lam_ckpt.keys():
        new_ckpt[key.replace("lam.", "")] = lam_ckpt[key]

    latent_action_model.load_state_dict(new_ckpt, strict=True)
    latent_action_model = latent_action_model.to(device_id).eval()
    
    # 1. 데이터 로더 호출 부분
    dataloader, stats = load_data_manual(
        dataset_dir=cfg.data_root_dir,
        batch_size=cfg.batch_size,
        processor=processor,
        window_size=cfg.window_size
    )

    # 2. 통계 저장 부분 (state 통계도 저장하도록 수정)
    if distributed_state.is_main_process:
        stats_path = run_dir / 'dataset_statistics.json' # <-- .json으로 변경
        print(f"Saving dataset statistics to {stats_path}")
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=4) # <-- json.dump 사용

    wrapped_model, latent_action_model, optimizer, scheduler, dataloader = accelerator.prepare(
        wrapped_model, latent_action_model, optimizer, scheduler, dataloader
    )

    # Deque to store recent train metrics (used for computing smoothened metrics for gradient accumulation)
    recent_losses = deque(maxlen=cfg.grad_accumulation_steps)
    recent_action_accuracies = deque(maxlen=cfg.grad_accumulation_steps)
    recent_l1_losses = deque(maxlen=cfg.grad_accumulation_steps)

    # Train!
    with tqdm.tqdm(total=cfg.max_steps, leave=False) as progress:
        wrapped_model.train()
        optimizer.zero_grad()
        current_step = 0
        while current_step < cfg.max_steps:
            for batch_idx, batch in enumerate(dataloader):
                batch["initial_pixel_values"] = batch["initial_pixel_values"].to(device_id)
                batch["target_pixel_values"] = batch["target_pixel_values"].to(device_id)
                batch["pixel_values"] = batch["pixel_values"].to(torch.bfloat16).to(device_id)
                batch['actions'] = batch['actions'].to(device_id)
                batch['proprio'] = batch['proprio'].to(device_id)

                ### [TODO] We construct latent action labels (also history latent actions) on-the-fly
                ### This is a work-round of potential CUDA conflict of calling models in dataloader
                if len(batch["initial_pixel_values_hist"]) > 1:
                    batch["initial_pixel_values_hist"] = batch["initial_pixel_values_hist"].to(device_id)
                    batch["target_pixel_values_hist"] = batch["target_pixel_values_hist"].to(device_id)

                    with torch.no_grad():
                        video = torch.stack([batch["initial_pixel_values"], batch["target_pixel_values"]], dim=1)
                        latent_action_idx_batch = latent_action_model.module.vq_encode(video)['indices'].squeeze()
                        video = torch.stack([batch["initial_pixel_values_hist"], batch["target_pixel_values_hist"]], dim=1)
                        latent_action_idx_history = latent_action_model.module.vq_encode(video)['indices'].squeeze()

                    input_ids_list = []
                    labels_list = []
                    hist_idx = 0
                    # print(batch['with_hist'],latent_action_idx_history.shape)
                    for idx, latent_action_idx in enumerate(latent_action_idx_batch):
                        action_vocab = [f'<ACT_{i.item()}>' for i in latent_action_idx]   # [ACT_1, ACT_2, ... ACT_K]
                        action_tokens = ''
                        for i, action in enumerate(action_vocab):
                            action_tokens += action
                        
                        if batch['with_hist'][idx]:
                            action_vocab = [f'<ACT_{i.item()}>' for i in latent_action_idx_history[hist_idx]]

                            hist_action_tokens = ''
                            for i, action in enumerate(action_vocab):
                                hist_action_tokens += action

                            input_prompt = f"What action should the robot take to {batch['instructions'][idx].lower()}? History action " + hist_action_tokens
                            hist_idx += 1
                        else:
                            input_prompt = f"What action should the robot take to {batch['instructions'][idx].lower()}?"


                        # Add instruction to VLA prompt
                        prompt_builder = PurePromptBuilder("openvla")
                        conversation = [
                            {"from": "human", "value": input_prompt},
                            {"from": "gpt", "value": action_tokens},
                        ]
                        for turn in conversation:
                            prompt_builder.add_turn(turn["from"], turn["value"])

                        # Tokenize (w/ `base_tokenizer`)
                        input_ids = processor.tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
                        labels = list(input_ids)

                        # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
                        #   =>> IMPORTANT :: IF WE'RE USING HF .forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
                        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)

                        labels[: -(len(action_vocab) + 1)] = -100

                        input_ids_list.append(input_ids)
                        labels_list.append(labels)
                
                else:
                    with torch.no_grad():
                        video = torch.stack([batch["initial_pixel_values"], batch["target_pixel_values"]], dim=1)
                        latent_action_idx_batch = latent_action_model.module.vq_encode(video)['indices'].squeeze()

                    input_ids_list = []
                    labels_list = []
                    for idx, latent_action_idx in enumerate(latent_action_idx_batch):
                        action_vocab = [f'<ACT_{i.item()}>' for i in latent_action_idx]   # [ACT_1, ACT_2, ... ACT_K]

                        action_tokens = ''
                        for i, action in enumerate(action_vocab):
                            action_tokens += action

                        # Add instruction to VLA prompt
                        prompt_builder = PurePromptBuilder("openvla")
                        conversation = [
                            {"from": "human", "value": f"What action should the robot take to {batch['instructions'][idx].lower()}?"},
                            {"from": "gpt", "value": action_tokens},
                        ]
                        for turn in conversation:
                            prompt_builder.add_turn(turn["from"], turn["value"])

                        # Tokenize (w/ `base_tokenizer`)
                        input_ids = processor.tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
                        labels = list(input_ids)

                        # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
                        #   =>> IMPORTANT :: IF WE'RE USING HF .forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
                        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)

                        labels[: -(len(action_vocab) + 1)] = -100

                        input_ids_list.append(input_ids)
                        labels_list.append(labels)

                input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=processor.tokenizer.pad_token_id)
                labels = pad_sequence(labels_list, batch_first=True, padding_value=-100)

                # Truncate (if necessary)
                input_ids, labels = input_ids[:, : processor.tokenizer.model_max_length], labels[:, : processor.tokenizer.model_max_length]

                # Get `attention_mask` by checking for `pad_token_id`
                attention_mask = input_ids.ne(processor.tokenizer.pad_token_id)

                batch["input_ids"] = input_ids
                batch["attention_mask"] = attention_mask
                batch["labels"] = labels

                output, act_loss, loss_one_step, latent_action_tokens = wrapped_model(batch)

                loss = act_loss if cfg.freeze_vla else act_loss + output.loss
                # Normalize loss to account for gradient accumulation
                normalized_loss = loss / cfg.grad_accumulation_steps

                torch.nn.utils.clip_grad_norm_(wrapped_model.parameters(), max_norm=0.3)
                # Backward pass
                normalized_loss.backward()

                # Compute Accuracy and L1 Loss for Logging
                action_logits = output.logits[:, wrapped_model.module.vla.vision_backbone.featurizer.patch_embed.num_patches : -1]
                action_preds = action_logits.argmax(dim=2)
                action_gt = batch["labels"][:, 1:].to(action_preds.device)
                mask = action_gt > 32000

                # Compute Accuracy
                correct_preds = (action_preds == action_gt) & mask
                action_accuracy = correct_preds.sum().float() / mask.sum().float()

                # Store recent train metrics
                recent_losses.append(loss.item())
                recent_action_accuracies.append(action_accuracy.item())

                # Compute gradient step index
                gradient_step_idx = batch_idx // cfg.grad_accumulation_steps

                # Compute smoothened train metrics
                #   =>> Equal to current step metrics when not using gradient accumulation
                #   =>> Otherwise, equal to the average of metrics observed over micro-batches used for gradient accumulation
                smoothened_loss = sum(recent_losses) / len(recent_losses)
                smoothened_action_accuracy = sum(recent_action_accuracies) / len(recent_action_accuracies)

                # Optimizer Step
                if (batch_idx + 1) % cfg.grad_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    progress.update()

                # Save Model Checkpoint
                if (gradient_step_idx + current_step) > 0 and (gradient_step_idx + current_step) % cfg.save_steps == 0:
                    step = gradient_step_idx + current_step
                    print(f"Process {distributed_state.process_index}: Reached checkpoint step {step}.")

                    # 이 체크포인트를 위한 경로들을 명확히 정의합니다.
                    checkpoint_run_dir = run_dir / str(step)
                    checkpoint_adapter_dir = adapter_dir / str(step)

                    # 메인 프로세스만 파일 쓰기 작업을 수행합니다.
                    if distributed_state.is_main_process: 
                        print(f"Saving Model Checkpoint for Step {step}")

                        # ★★★핵심 수정: LoRA 어댑터를 저장할 폴더를 명시적으로 생성합니다.
                        os.makedirs(checkpoint_run_dir, exist_ok=True)
                        if cfg.use_lora:
                            os.makedirs(checkpoint_adapter_dir, exist_ok=True)
                            
                        # LoRA 어댑터 가중치를 임시 어댑터 폴더에 저장합니다.
                        if cfg.use_lora and not cfg.freeze_vla:
                            wrapped_model.module.vla.save_pretrained(checkpoint_adapter_dir)
                            
                        # Action Decoder 가중치를 메인 결과 폴더에 저장합니다.
                        decoder_save_path = checkpoint_run_dir / f'action_decoder-{step}.pt'
                        torch.save(wrapped_model.module.action_decoder.state_dict(), decoder_save_path)
                        
                        # 프로세서 설정 파일도 메인 결과 폴더에 저장합니다.
                        processor.save_pretrained(checkpoint_run_dir)

                    # 모든 프로세스가 메인 프로세스의 저장이 끝날 때까지 기다립니다.
                    dist.barrier()

                    # LoRA 가중치를 기본 모델과 병합하여 전체 모델 체크포인트를 저장합니다.
                    if cfg.use_lora:
                        # 모든 프로세스가 기본 VLA 모델을 다시 불러옵니다.
                        base_vla = AutoModelForVision2Seq.from_pretrained(
                            cfg.vla_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
                        )
                        
                        # 방금 저장한 어댑터 가중치를 불러와서 병합합니다.
                        merged_vla = PeftModel.from_pretrained(base_vla, checkpoint_adapter_dir)
                        merged_vla = merged_vla.merge_and_unload()
                        
                        if distributed_state.is_main_process:
                            # 병합된 전체 모델을 메인 결과 폴더에 저장합니다.
                            merged_vla.save_pretrained(checkpoint_run_dir)
                            print(f"Saved Merged Model Checkpoint for Step {step} at: {checkpoint_run_dir}")

                    # 모든 프로세스가 병합 및 저장이 끝날 때까지 기다립니다.
                    dist.barrier()

            description = f"Epoch {current_step + 1} | action_loss: {act_loss.item():.4f} | acc: {smoothened_action_accuracy:.4f}"
            progress.set_description(description)

            current_step = gradient_step_idx + 1 + current_step
            # Stop training when max_steps is reached
            if current_step >= cfg.max_steps:
                print(f"Max step {cfg.max_steps} reached! Stopping training...")
                wandb.finish()
                break


if __name__ == "__main__":
    # torch.multiprocessing.set_start_method('spawn', force=True)
    finetune()
