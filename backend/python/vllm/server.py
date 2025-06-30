import os
import asyncio
from concurrent import futures
import signal

from utils.logger import get_logger

import grpc
import proto.backend_pb2_grpc as backend_pb2_grpc
import proto.backend_pb2 as backend_pb2

MAX_WORKERS = int(os.environ.get("PYTHON_GRPC_MAX_WORKERS", "1"))

grpc_logger = get_logger("grpc_server")


class BackendServicer(backend_pb2_grpc.BackendServicer):
    def LoadModel(self, request, context):
        grpc_logger.info("Loading model...")
        return backend_pb2.GenerateTextResponse(text="Model loaded successfully")
        # ...

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
        futures.ThreadPoolExecutor(max_workers=MAX_WORKERS), options=[]
    )

    backend_pb2_grpc.add_BackendServicer_to_server(BackendServicer(), server)
    server.add_insecure_port(address)

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, server.stop, 0)

    await server.start()

    grpc_logger.info(f"gRPC server started on {address}")
