import os
import asyncio
from concurrent import futures
import signal
import argparse

from utils.logger import get_logger

import grpc
import backend_pb2_grpc
import backend_pb2

MAX_WORKERS = int(os.environ.get("PYTHON_GRPC_MAX_WORKERS", "1"))

grpc_logger = get_logger("grpc_server")


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

    def LoadModelShard(self, request, context):
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

            model_options = request.model_options
            llm_kwargs = {
                "model": request.model,
                "dtype": model_options.dtype or "auto",
                "enforce_eager": model_options.enforce_eager or False,
            }
        except Exception as e:

        return backend_pb2.GenerateTextResponse(text="Model loaded successfully")

    def Health(self, request, context):
        grpc_logger.info("Checking health...")
        return backend_pb2.HealthResponse(healthy=True, message="Service is healthy")
        # ...

    def GenerateText(self, request, context):
        # not completed
        # sampling_params = SamplingParams(
        #     temperature=request.temperature,
        #     top_p=request.top_p,
        #     max_tokens=request.max_tokens,
        # )
        # response = self.model.generate(request.prompt, sampling_params)
        return backend_pb2.GenerateTextResponse(text="Random sample text")

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
