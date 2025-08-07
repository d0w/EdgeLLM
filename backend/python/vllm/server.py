import os
import asyncio
from concurrent import futures
import signal
import argparse
import threading
from typing import Optional, Dict

from utils.logger import get_logger
from vllm import LLM, SamplingParams
from vllm.model_executor.parallel_utils.parallel_state import initialize_model_parallel

import grpc
import backend_pb2_grpc
import backend_pb2

MAX_WORKERS = int(os.environ.get("PYTHON_GRPC_MAX_WORKERS", "1"))

grpc_logger = get_logger("grpc_server")

# Get node ID from environment or generate one
node_id = os.environ.get("NODE_ID", f"node_{os.getpid()}")

class BackendServicer(backend_pb2_grpc.BackendServicer):
    def __init__(self):
        self.node_id = node_id
        self.llm: Optional[LLM] = None
        self.shard_info: Optional[Dict] = None
        self.shard_groups: Dict[str, Dict] = {}
        self.peer_clients: Dict[str, grpc.Channel] = {}
        self.active_requests: Dict[str, bool] = {}
        self._lock = threading.Lock()

        grpc_logger.info(f"BackendServicer initialized with node_id: {self.node_id}")

    def LoadModel(self, request, context):
        """Load a model for standard inference"""
        try:
            with self._lock:
                grpc_logger.info(f"Loading model: {request.model}")
                
                # Initialize model parallel if needed
                if request.tensor_parallel_size > 1:
                    initialize_model_parallel(
                        tensor_model_parallel_size=request.tensor_parallel_size,
                        pipeline_model_parallel_size=request.pipeline_parallel_size or 1
                    )
                
                # Create LLM instance
                self.llm = LLM(
                    model=request.model,
                    dtype=request.dtype or "auto",
                    tensor_parallel_size=request.tensor_parallel_size or 1,
                    pipeline_parallel_size=request.pipeline_parallel_size or 1,
                    max_model_len=request.max_model_len or None,
                    gpu_memory_utilization=request.gpu_memory_utilization or 0.9,
                    enforce_eager=request.enforce_eager or False,
                )
                
                grpc_logger.info(f"Model {request.model} loaded successfully")
                
                return backend_pb2.LoadModelResponse(
                    message="Model loaded successfully",
                    success=True,
                    model_id=request.model,
                    supported_features=["text_generation", "streaming"]
                )
                
        except Exception as e:
            grpc_logger.error(f"Failed to load model: {str(e)}")
            return backend_pb2.LoadModelResponse(
                message=f"Failed to load model: {str(e)}",
                success=False,
                model_id=""
            )

    def LoadModelShard(self, request, context):
        """Load a model shard for distributed inference"""
        try:
            grpc_logger.info(f"Loading model shard: {request.model}")
            assignment = request.assignment
            shard_rank = assignment.shard_rank
            world_size = assignment.world_size

            self.shard_info = {
                "model": request.model,
                "shard_group_id": request.shard_group_id,
                "shard_rank": shard_rank,
                "world_size": world_size,
                "layer_range": list(assignment.layer_range),
                "shard_type": assignment.shard_type,
                "metadata": dict(assignment.shard_metadata)
            }        

            # Initialize model parallel for sharding
            initialize_model_parallel(
                tensor_model_parallel_size=world_size if assignment.shard_type == "tensor_parallel" else 1,
                pipeline_model_parallel_size=world_size if assignment.shard_type == "pipeline" else 1
            )

            model_options = request.model_options
            llm_kwargs = {
                "model": request.model,
                "dtype": model_options.dtype or "auto",
                "enforce_eager": model_options.enforce_eager or False,
                "tensor_parallel_size": world_size if assignment.shard_type == "tensor_parallel" else 1,
                "pipeline_parallel_size": world_size if assignment.shard_type == "pipeline" else 1,
            }
            
            # Load the sharded model
            self.llm = LLM(**llm_kwargs)
            
            grpc_logger.info(f"Model shard loaded successfully for group {request.shard_group_id}")
            
            return backend_pb2.ShardLoadResponse(
                success=True,
                message="Shard loaded successfully",
                shard_id=f"{request.shard_group_id}_{shard_rank}",
                num_layers=assignment.layer_range[1] - assignment.layer_range[0] if assignment.layer_range else 0
            )
            
        except Exception as e:
            grpc_logger.error(f"Failed to load model shard: {str(e)}")
            return backend_pb2.ShardLoadResponse(
                success=False,
                message=f"Failed to load shard: {str(e)}",
                shard_id="",
                num_layers=0
            )

    def Health(self, request, context):
        """Health check with detailed status"""
        try:
            # Check if model is loaded
            model_loaded = self.llm is not None
            
            # Get basic system info
            active_requests = len(self.active_requests)
            loaded_shards = [self.shard_info["shard_group_id"]] if self.shard_info else []
            
            grpc_logger.info("Health check completed")
            return backend_pb2.HealthResponse(
                healthy=model_loaded,
                message="Service is healthy" if model_loaded else "No model loaded",
                load_percentage=min(100, active_requests * 10),  # Rough estimate
                active_requests=active_requests,
                loaded_shards=loaded_shards
            )
        except Exception as e:
            grpc_logger.error(f"Health check failed: {str(e)}")
            return backend_pb2.HealthResponse(
                healthy=False,
                message=f"Health check failed: {str(e)}",
                active_requests=0
            )

    def GenerateText(self, request, context):
        """Generate text using the loaded model"""
        try:
            if self.llm is None:
                return backend_pb2.GenerateTextResponse(
                    text="",
                    finished=True,
                    finish_reason="error: no model loaded",
                    request_id=request.request_id
                )
            
            # Track active request
            request_id = request.request_id or f"req_{len(self.active_requests)}"
            self.active_requests[request_id] = True
            
            try:
                # Create sampling parameters
                sampling_params = SamplingParams(
                    temperature=request.temperature or 0.7,
                    top_p=request.top_p or 1.0,
                    top_k=int(request.top_k) if request.top_k > 0 else -1,
                    max_tokens=request.max_tokens or 100,
                    stop=list(request.stop) if request.stop else None,
                    frequency_penalty=request.frequency_penalty or 0.0,
                    presence_penalty=request.presence_penalty or 0.0,
                    seed=int(request.seed) if request.seed > 0 else None
                )
                
                # Generate text
                grpc_logger.info(f"Generating text for request {request_id}")
                outputs = self.llm.generate([request.prompt], sampling_params)
                
                if outputs and len(outputs) > 0:
                    output = outputs[0]
                    generated_text = output.outputs[0].text if output.outputs else ""
                    finish_reason = output.outputs[0].finish_reason if output.outputs else "unknown"
                    
                    # Calculate token counts (rough estimate)
                    prompt_tokens = len(request.prompt.split())
                    completion_tokens = len(generated_text.split())
                    
                    return backend_pb2.GenerateTextResponse(
                        text=generated_text,
                        finished=True,
                        finish_reason=finish_reason,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=prompt_tokens + completion_tokens,
                        request_id=request_id
                    )
                else:
                    return backend_pb2.GenerateTextResponse(
                        text="",
                        finished=True,
                        finish_reason="no_output",
                        request_id=request_id
                    )
                    
            finally:
                # Remove from active requests
                self.active_requests.pop(request_id, None)
                
        except Exception as e:
            grpc_logger.error(f"Text generation failed: {str(e)}")
            return backend_pb2.GenerateTextResponse(
                text="",
                finished=True,
                finish_reason=f"error: {str(e)}",
                request_id=request.request_id
            )

    # ... more functions


async def serve(address):
    server = grpc.aio.server(
        futures.ThreadPoolExecutor(max_workers=MAX_WORKERS),
        options=[
            ("grpc.max_message_length", 50 * 1024 * 1024),  # 50MB
            ("grpc.max_send_message_length", 50 * 1024 * 1024),  # 50MB
            ("grpc.max_receive_message_length", 50 * 1024 * 1024),  # 50MB
        ],
    )

    backend_pb2_grpc.add_BackendServicer_to_server(BackendServicer(), server)
    server.add_insecure_port(address)

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.ensure_future(server.stop(5)))

    await server.start()

    grpc_logger.info(f"gRPC server started on {address}")

    await server.wait_for_termination()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the gRPC server.")
    parser.add_argument(
        "--addr", default="localhost:50051", help="The address to bind the server to."
    )
    args = parser.parse_args()

    asyncio.run(serve(args.addr))
