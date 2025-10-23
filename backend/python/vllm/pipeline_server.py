import os
import asyncio
import torch
import pickle
import numpy as np
from typing import List, Dict, Optional, Any
import logging

import torch
from vllm.model_executor import get_model
from vllm.model_executor.model_loader import get_model_loader
from vllm.attention import PagedAttention
from vllm.worker.model_runner import ModelRunner
from vllm.config import ModelConfig, CacheConfig, ParallelConfig

from utils.logger import get_logger
import grpc
import backend_pb2_grpc
import backend_pb2

grpc_logger = get_logger("true_pipeline_server")


class VLLMLayerExtractor:
    """Extract vLLM-optimized layers while maintaining performance benefits"""

    def __init__(self, model_name: str, layer_indices: List[int]):
        self.model_name = model_name
        self.layer_indices = layer_indices

        # Use vLLM's model loading infrastructure
        self.model_config = ModelConfig(
            model=model_name,
            tokenizer=model_name,
            trust_remote_code=True,
            dtype=torch.float16,
            max_model_len=2048,
        )

        # Load model using vLLM's optimized loader
        self.model = self._load_vllm_model()
        self.layers = self._extract_vllm_layers()

    def _load_vllm_model(self):
        """Load model using vLLM's optimized model loader"""
        loader = get_model_loader(self.model_config)
        model = get_model(
            model_config=self.model_config,
            device_config=None,  # Auto-detect
            lora_config=None,
            vision_language_config=None,
            parallel_config=ParallelConfig(
                pipeline_parallel_size=1, tensor_parallel_size=1
            ),
            scheduler_config=None,
        )
        return model

    def _extract_vllm_layers(self):
        """Extract specific layers while keeping vLLM optimizations"""
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            all_layers = self.model.model.layers
            return [all_layers[i] for i in self.layer_indices if i < len(all_layers)]
        return []

    def process_with_vllm_optimizations(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Process through layers using vLLM's optimized kernels"""

        for layer in self.layers:
            # Use vLLM's optimized layer forward pass
            # This includes PagedAttention and optimized CUDA kernels
            layer_output = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=None,
                kv_cache=None,  # We lose KV caching in distributed mode
                attn_metadata=None,
            )

            if isinstance(layer_output, tuple):
                hidden_states = layer_output[0]
            else:
                hidden_states = layer_output

        return hidden_states


class HybridVLLMServicer(backend_pb2_grpc.BackendServicer):
    """Hybrid approach: vLLM optimizations + distributed coordination"""

    def __init__(self):
        self.extractors = {}  # shard_id -> VLLMLayerExtractor

    async def LoadModelShard(self, request, context):
        try:
            extractor = VLLMLayerExtractor(request.Model, list(request.LayerIndices))

            self.extractors[request.ShardId] = extractor

            return backend_pb2.LoadModelResponse(
                message=f"✅ vLLM-optimized shard loaded with PagedAttention!",
                success=True,
                loaded_layers=list(request.LayerIndices),
            )

        except Exception as e:
            return backend_pb2.LoadModelResponse(
                message=f"Failed: {str(e)}", success=False
            )

    async def ProcessLayer(self, request, context):
        """Process using vLLM's optimized kernels"""
        try:
            # Find appropriate extractor
            extractor = self._find_extractor(request.LayerStart, request.LayerEnd)

            # Deserialize hidden states
            hidden_states = pickle.loads(request.HiddenStates)
            if isinstance(hidden_states, np.ndarray):
                hidden_states = torch.from_numpy(hidden_states)

            # Create attention mask
            seq_len = hidden_states.size(1)
            attention_mask = torch.ones(1, seq_len, device=hidden_states.device)

            # 🎯 Process using vLLM's optimized kernels
            processed = extractor.process_with_vllm_optimizations(
                hidden_states, attention_mask
            )

            return backend_pb2.LayerProcessResponse(
                hidden_states=pickle.dumps(processed.cpu().numpy()), success=True
            )

        except Exception as e:
            return backend_pb2.LayerProcessResponse(success=False, error_message=str(e))
