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
