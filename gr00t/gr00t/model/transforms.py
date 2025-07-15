# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random
import re
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import tree
from einops import rearrange
from PIL import Image
from pydantic import Field, PrivateAttr
from transformers import AutoProcessor, ProcessorMixin
from transformers.data.data_collator import DataCollatorMixin
from transformers.feature_extraction_utils import BatchFeature

from gr00t.data.embodiment_tags import EMBODIMENT_TAG_MAPPING, EmbodimentTag
from gr00t.data.schema import DatasetMetadata
from gr00t.data.transform.base import InvertibleModalityTransform

from .backbone.eagle_backbone import DEFAULT_EAGLE_PATH


def formalize_language(language: str) -> str:
    """
    1. Force lowercase
    2. Remove all punctuations
    """
    language = language.lower()
    language = re.sub(r"[^\w\s]", "", language)
    return language


def build_eagle_processor(eagle_path: str) -> ProcessorMixin:
    eagle_processor = AutoProcessor.from_pretrained(
        eagle_path, trust_remote_code=True, use_fast=True
    )
    eagle_processor.tokenizer.padding_side = "left"
    return eagle_processor


def collate(features: List[dict], eagle_processor) -> dict:
    batch = {}
    keys = features[0].keys()

    for key in keys:
        values = [elem[key] for elem in features]

        if key == "eagle_content":
            # eagle_content is now handled in _apply_vlm_processing
            # Skip processing here as eagle_ keys are already added
            continue
        elif key.startswith("eagle_"):
            # Directly use the processed eagle inputs
            if key == "eagle_pixel_values":
                # Handle empty pixel_values case
                if all(v.numel() == 0 for v in values):
                    # All are empty, create a proper empty tensor
                    batch[key] = torch.empty(len(values), 0, 224, 224, dtype=torch.float32)
                else:
                    # Normal concatenation for non-empty tensors
                    batch[key] = torch.cat([v for v in values if v.numel() > 0])
            elif key == "eagle_image_sizes":
                # Handle empty image_sizes case
                if all(v.numel() == 0 for v in values):
                    # All are empty, create a proper empty tensor
                    batch[key] = torch.empty(len(values), 0, 2, dtype=torch.long)
                else:
                    # Normal concatenation for non-empty tensors
                    batch[key] = torch.cat([v for v in values if v.numel() > 0])
            elif key in ("eagle_input_ids", "eagle_attention_mask"):
                # Handle sequence data that may have different lengths
                # Apply padding to make all sequences the same length
                max_length = max(v.shape[-1] for v in values)
                padded_values = []
                for v in values:
                    if v.shape[-1] < max_length:
                        # Pad the sequence
                        if key == "eagle_input_ids":
                            # Use pad_token_id for input_ids (typically 0)
                            pad_value = getattr(eagle_processor.tokenizer, 'pad_token_id', 0)
                        else:
                            # Use 0 for attention_mask
                            pad_value = 0
                        
                        padding_size = max_length - v.shape[-1]
                        if v.ndim == 1:
                            padded_v = torch.cat([v, torch.full((padding_size,), pad_value, dtype=v.dtype, device=v.device)])
                        else:
                            padding_shape = list(v.shape)
                            padding_shape[-1] = padding_size
                            padded_v = torch.cat([v, torch.full(padding_shape, pad_value, dtype=v.dtype, device=v.device)], dim=-1)
                        padded_values.append(padded_v)
                    else:
                        padded_values.append(v)
                batch[key] = torch.cat(padded_values)
            else:
                # For other eagle keys, concatenate normally
                batch[key] = torch.cat(values)
        elif key in ("pixel_values", "image_grid_thw", "attention_mask", "input_ids"):
            # Concat in existing batch dimension.
            batch[key] = torch.cat(values)
        else:
            # state, state_mask, action and action_mask.
            # Stack to form the batch dimension.
            if isinstance(values[0], torch.Tensor):
                # Already tensors, just stack
                batch[key] = torch.stack(values)
            else:
                # Numpy arrays, convert to tensor then stack
                batch[key] = torch.from_numpy(np.stack(values))
    return batch


class DefaultDataCollator(DataCollatorMixin):
    def __init__(self, eagle_path: str = DEFAULT_EAGLE_PATH):
        super().__init__()
        self.eagle_processor = build_eagle_processor(eagle_path)

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        return collate(features, self.eagle_processor)


class GR00TTransform(InvertibleModalityTransform):

    # -- We inherit from ModalityTransform, so we keep apply_to as well --
    apply_to: list[str] = Field(
        default_factory=list, description="Not used in this transform, kept for compatibility."
    )
    training: bool = Field(
        default=True, description="Whether to apply the transform in training mode."
    )
    formalize_language: bool = Field(default=False, description="Formalize language if True.")
    embodiment_tag_mapping: dict[str, int] = Field(
        description="The projector index of each embodiment tag.",
        default=EMBODIMENT_TAG_MAPPING,
    )
    language_dropout_prob: float = Field(
        default=0.0,
        description="Dropout probability for language.",
    )

    # Private attributes to keep track of shapes/dimensions across apply/unapply
    _language_key: Optional[list[str]] = PrivateAttr(default=None)

    eagle_processor: ProcessorMixin = Field(default=build_eagle_processor(DEFAULT_EAGLE_PATH))

    # XEmbDiT arguments
    default_instruction: str = Field(default="Perform the default behavior.")
    max_state_dim: int
    max_action_dim: int
    state_horizon: int
    action_horizon: int

    max_length: int = 512
    embodiment_tag: EmbodimentTag | None = None

    def set_metadata(self, dataset_metadata: DatasetMetadata):
        """Set the metadata for the transform."""
        super().set_metadata(dataset_metadata)
        self.embodiment_tag = dataset_metadata.embodiment_tag

    def get_embodiment_tag(self) -> int:
        """Get the embodiment tag from the data."""
        assert (
            self.embodiment_tag is not None
        ), "Embodiment tag not set. Please call set_metadata first."
        return self.embodiment_tag_mapping[self.embodiment_tag.value]

    def check_keys_and_batch_size(self, data):
        grouped_keys = {}
        for key in data.keys():
            if "annotation" in key:
                modality = "language"
            else:
                try:
                    modality, _ = key.split(".")
                except:  # noqa: E722
                    modality = "others"  # will contain the video, state, and action
            if modality not in grouped_keys:
                grouped_keys[modality] = []
            grouped_keys[modality].append(key)
        
        # Check if video exists, if not assume single batch
        if "video" in data:
            # Use video key to determine batch size.
            video_ndim = data["video"].ndim
            if video_ndim == 5:  # Interpret as [T, V, H, W, C]
                is_batched = False
                batch_size = 1
            elif video_ndim == 6:  # Interpret as [B, T, V, H, W, C]
                is_batched = True
                batch_size = data["video"].shape[0]
            else:
                raise ValueError(f"Unsupported video number of dimensions: {video_ndim}")
        else:
            # No video data, check other modalities for batch size
            # Use state or action to determine batch size
            if "state" in data:
                state_ndim = data["state"].ndim
                if state_ndim == 2:  # [T, D]
                    is_batched = False
                    batch_size = 1
                elif state_ndim == 3:  # [B, T, D]
                    is_batched = True
                    batch_size = data["state"].shape[0]
                else:
                    raise ValueError(f"Unsupported state number of dimensions: {state_ndim}")
            elif "action" in data:
                action_ndim = data["action"].ndim
                if action_ndim == 2:  # [T, D]
                    is_batched = False
                    batch_size = 1
                elif action_ndim == 3:  # [B, T, D]
                    is_batched = True
                    batch_size = data["action"].shape[0]
                else:
                    raise ValueError(f"Unsupported action number of dimensions: {action_ndim}")
            else:
                # Default to single batch if no recognizable modality
                is_batched = False
                batch_size = 1

        # Handle language
        if "language" in grouped_keys:
            language_keys = grouped_keys["language"]
            assert len(language_keys) == 1, f"{language_keys=}"
            self._language_key = language_keys[0]
        return is_batched, batch_size

    def _apply_vlm_processing(self, batch: dict) -> BatchFeature:
        """
        Args:
            batch:
                images: [V, T, C, H, W] (optional)
                language: str
        Returns: required input with the format `BatchFeature`
        """
        text_content = []

        # handle language
        lang = batch["language"]
        if isinstance(lang, list):
            lang = lang[0]
        text_content.append({"type": "text", "text": lang})

        # Handle images if available
        if "images" in batch:
            images = batch["images"]  # [V, T, C, H, W]
            np_images = rearrange(images, "v t c h w -> (t v) c h w")
            eagle_images = [Image.fromarray(np.transpose(v, (1, 2, 0))) for v in np_images]
            eagle_image = [{"type": "image", "image": img} for img in eagle_images]
            eagle_conversation = [
                {
                    "role": "user",
                    "content": eagle_image + text_content,
                }
            ]
            image_inputs, video_inputs = self.eagle_processor.process_vision_info(eagle_conversation)
        else:
            # No images, only text
            eagle_conversation = [
                {
                    "role": "user",
                    "content": text_content,
                }
            ]
            # Set empty lists for image/video inputs when no images
            image_inputs = []
            video_inputs = []

        text_list = [
            self.eagle_processor.apply_chat_template(
                eagle_conversation, tokenize=False, add_generation_prompt=True
            )
        ]
        
        # Process with eagle_processor to get proper tensor format
        if image_inputs:
            # Normal case with images
            eagle_inputs = self.eagle_processor(
                text=text_list, images=image_inputs, return_tensors="pt", padding=True
            )
        else:
            # No images case - create empty pixel_values and image_sizes
            eagle_inputs = self.eagle_processor(
                text=text_list, images=None, return_tensors="pt", padding=True
            )
            # Add empty pixel_values and image_sizes that Eagle model expects
            eagle_inputs["pixel_values"] = torch.empty(0, 0, 224, 224, dtype=torch.float32)
            eagle_inputs["image_sizes"] = torch.empty(0, 2, dtype=torch.long)
        
        eagle_content = {
            "image_inputs": image_inputs,
            "video_inputs": video_inputs,
            "text_list": text_list,
        }
        inputs = {}
        inputs["eagle_content"] = eagle_content
        
        # Add eagle_inputs directly to inputs with eagle_ prefix
        for k, v in eagle_inputs.items():
            inputs["eagle_" + k] = v
            
        return inputs

    def _prepare_video(self, data: dict):
        """Process, stack, and pad images from data['video']. Returns None if no video."""
        if "video" not in data:
            return None
            
        images = rearrange(
            data["video"],
            "t v h w c -> v t c h w",
        )
        return images

    def _prepare_language(self, data: dict):
        """Tokenize data['language'] (or default_instruction if missing)."""
        if self._language_key is not None:
            raw_language = data[self._language_key]
            if isinstance(raw_language, list):
                raw_language = raw_language[0]

            # Language dropout
            if self.training and self.language_dropout_prob > 1e-9:
                if random.random() < self.language_dropout_prob:
                    raw_language = self.default_instruction
        else:
            raw_language = self.default_instruction
        return raw_language

    def _prepare_state(self, data: dict):
        """
        Gathers final state from data['state'], then pads to max_state_dim.
        Return (state, state_mask, n_state_tokens).
        """
        if "state" not in data:
            state = np.zeros((self.state_horizon, self.max_state_dim))
            state_mask = np.zeros((self.state_horizon, self.max_state_dim), dtype=bool)
            n_state_tokens = self.state_horizon
            return state, state_mask, n_state_tokens

        state = data["state"]
        
        # Check if state is batched (3D) or single (2D)
        if state.ndim == 3:  # [B, T, D] - batched
            batch_size, time_steps, state_dims = state.shape
            assert time_steps == self.state_horizon, f"{state.shape=}, {self.state_horizon=}"
            
            n_state_dims = state_dims
            
            # Handle padding for batched data
            if n_state_dims > self.max_state_dim:
                state = state[:, :, :self.max_state_dim]
                n_state_dims = self.max_state_dim
            else:
                # Pad up to max_state_dim if smaller - 3D padding
                state = np.pad(state, ((0, 0), (0, 0), (0, self.max_state_dim - n_state_dims)), "constant")
            
            # Create mask for real state dims
            state_mask = np.zeros_like(state).astype(bool)
            state_mask[:, :, :n_state_dims] = True
            
        elif state.ndim == 2:  # [T, D] - single sample
            time_steps, state_dims = state.shape
            assert time_steps == self.state_horizon, f"{state.shape=}, {self.state_horizon=}"
            
            n_state_dims = state_dims
            
            # Handle padding for single sample
            if n_state_dims > self.max_state_dim:
                state = state[:, :self.max_state_dim]
                n_state_dims = self.max_state_dim
            else:
                # Pad up to max_state_dim if smaller - 2D padding
                state = np.pad(state, ((0, 0), (0, self.max_state_dim - n_state_dims)), "constant")
            
            # Create mask for real state dims
            state_mask = np.zeros_like(state).astype(bool)
            state_mask[:, :n_state_dims] = True
        else:
            raise ValueError(f"Unsupported state dimensions: {state.ndim}. Expected 2D [T, D] or 3D [B, T, D]")

        # We only have 1 "proprio" token to represent the entire state
        n_state_tokens = state.shape[-2]  # Use -2 to get time dimension for both 2D and 3D
        return state, state_mask, n_state_tokens

    def _prepare_action(self, data: dict):
        """
        Pad to max_action_dim, return masks.
        """
        if "action" not in data:
            actions = np.zeros((self.action_horizon, self.max_action_dim))
            actions_mask = np.zeros((self.action_horizon, self.max_action_dim), dtype=bool)
            n_action_tokens = self.action_horizon
            return actions, actions_mask, n_action_tokens

        actions = data["action"]
        
        # Check if actions is batched (3D) or single (2D)
        if actions.ndim == 3:  # [B, T, D] - batched
            batch_size, time_steps, action_dims = actions.shape
            assert time_steps == self.action_horizon, f"{actions.shape=}, {self.action_horizon=}"
            
            n_action_tokens = time_steps
            n_action_dims = action_dims
            
            assert (
                n_action_dims <= self.max_action_dim
            ), f"Action dim {n_action_dims} exceeds max allowed {self.max_action_dim}."
            
            # Pad the channel dimension for batched data
            actions = np.pad(actions, ((0, 0), (0, 0), (0, self.max_action_dim - n_action_dims)), "constant")
            
            # Create mask: [B, T, max_action_dim]
            actions_mask = np.zeros((batch_size, n_action_tokens, self.max_action_dim), dtype=bool)
            actions_mask[:, :, :n_action_dims] = True
            
        elif actions.ndim == 2:  # [T, D] - single sample
            time_steps, action_dims = actions.shape
            assert time_steps == self.action_horizon, f"{actions.shape=}, {self.action_horizon=}"
            
            n_action_tokens = time_steps
            n_action_dims = action_dims
            
            assert (
                n_action_dims <= self.max_action_dim
            ), f"Action dim {n_action_dims} exceeds max allowed {self.max_action_dim}."
            
            # Pad the channel dimension for single sample
            actions = np.pad(actions, ((0, 0), (0, self.max_action_dim - n_action_dims)), "constant")
            
            # Create mask: [T, max_action_dim]
            actions_mask = np.zeros((n_action_tokens, self.max_action_dim), dtype=bool)
            actions_mask[:, :n_action_dims] = True
        else:
            raise ValueError(f"Unsupported action dimensions: {actions.ndim}. Expected 2D [T, D] or 3D [B, T, D]")

        return actions, actions_mask, n_action_tokens

    def apply_single(self, data: dict) -> dict:
        transformed_data = {}

        # 1) Prepare video and language with vlm processing.
        images = self._prepare_video(data)
        language = self._prepare_language(data)
        
        # Create batch data for VLM processing
        batch_data = {"language": language}
        if images is not None:
            images = images.astype(np.uint8)
            batch_data["images"] = images
            
        vlm_outputs = self._apply_vlm_processing(batch_data)

        # 2) Prepare state
        state, state_mask, _ = self._prepare_state(data)
        transformed_data["state"] = state
        transformed_data["state_mask"] = state_mask

        if self.training:
            # 3) Prepare actions
            transformed_data["segmentation_target"] = np.zeros((2,))
            transformed_data["segmentation_target_mask"] = np.zeros((1,))
            transformed_data["has_real_action"] = np.ones((), dtype=bool)
            actions, actions_mask, _ = self._prepare_action(data)
            transformed_data["action"] = actions
            transformed_data["action_mask"] = actions_mask

        for k, v in vlm_outputs.items():
            assert k not in transformed_data, f"Key {k} already exists in transformed_data."
            transformed_data[k] = v

        transformed_data["embodiment_id"] = self.get_embodiment_tag()

        if self.training:
            action_and_mask_keys = ["action", "action_mask"]
            assert all(
                transformed_data[key].shape == transformed_data["action"].shape
                for key in action_and_mask_keys
            ), f"Shape mismatch: {[(key, transformed_data[key].shape) for key in action_and_mask_keys]}"

        return transformed_data

    def apply_batch(self, data: dict, batch_size: int) -> dict:
        # Split on batch dimension.
        data_split = [tree.map_structure(lambda x: x[i], data) for i in range(batch_size)]
        # Process each element.
        data_split_processed = [self.apply_single(elem) for elem in data_split]
        return collate(data_split_processed, self.eagle_processor)

    def apply(self, data: dict) -> dict:
        is_batched, batch_size = self.check_keys_and_batch_size(data)
        if is_batched:
            return self.apply_batch(data, batch_size)
        else:
            return self.apply_single(data)

    def unapply(self, data: dict) -> dict:
        # Leave as is so that ConcatTransform can split the values
        return data

    def __call__(self, data: dict) -> dict:
        return self.apply(data)
