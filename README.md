# EdgeLLM

Run decentralized LLMs. Distribute your inference and training requests across a vast peer-to-peer network without a need for a centralized server. EdgeLLM gives users the ability to perform inference on models much larger than they could with their own machine, utilizing a network of volunteers' machines to perform inference on their behalf.

## Manifesto
OpenAI's ChatGPT, Google's Gemini, Anthropic's Claude. With their powerful models backed by their supercomputers, they're a great resource to have... until you start thinking what if one day they pull the plug? What if they begin charging insane prices? What happens to all the embarassing questions you have about your codebase (just me?). All of these LLM providers make it super easy to begin chatting with insanely complex models (even up to 1T parameters) but it comes at the cost of an increased reliance on their services and trusting them to handle your data and trust correctly. I, like many, will never put their full trust into something they do not personally own *entirely* and cannot see myself relying day-to-day on intelligent LLMs that are running on someone else's computer thousands of kilometers away with sketchy privacy policies at best. 

As we spammed ChatGPT, Claude, Gemini, and more with our mindless typing, we asked a question to ourselves. Why leave some of the most powerful technologies in the 21st century to the hands of a few's large investments in thousands of GPUs when so many individuals already have their own laptop and even a GPU? An individual might not be able to run Gemini Pro themselves but borrow 100 people's GPUs for a bit and it'll be possible. We see large P2P projects like BitTorrent already do this with filesharing, so let's do it with LLMs too. We think that as the internet should truly be, LLMs should be likewise: decentralized and powered by the people.

## Goals

1. Get a vLLM cluster up and running triggered by EdgeLLM requests (*we are here*)
2. Implement P2P connections
   - Involves adding standards for capabilities, metadata, etc. and how to propagate this around
3. Add join/leave network capabilities
4. KV Cache dissaggregation/share with IPFS
5. P/D Dissaggregation
6. Model hotswapping
7. More as we go along

## Features

- Peer-to-peer decentralized inference
- vLLM support
- Distributed KV Cache over P2P filesystem
- P/D Dissaggregation
- more to come...

## Implementation

*In prototype stage so subject to change*
The goal of this is a truly decentralized p2p inference server that allows interaction with large LLM models (100B+) may not be able to be run by individuals alone.

This is a wrapper on top of runners such as vLLM and llama.cpp that support distributed inference already. The key is that these do not support true peer-to-peer networking and either relies on a head node or other centralized methods of performing the inference.

We take these runners and add bindings for them, then wrap the bindings with our p2p node implementation. Each node will have the ability to host a model or perform inference. When a user wishes to start hosting a model, they will join the network with metadata about their hosting capabilities as well as what model to host. Using `go-libp2p`, we can use its DHT functionalities to achieve this.

Once users join the network, inference requests can now be handled. When a user joins the network as a "leecher", they will initiate an inference request that will create an object that handles which user requested it, the model, the prompt/response, and various metadata. Our code will then handle finding the optimal peers with a combined minimum availability (more details will need to be fleshed out for this) for that request. Once the workers are established, we then call the inference backends such as vLLM to create a one-time distributed inference network with the requester as the head and the workers from the network. The inference will be performed and then the vLLM/other backend server will be closed. There should also be the option to open a "session" and only close and open on a session.

Users hosting a model will need to have the entire model installed, but there needs to be more testing as to whether the whole model needs to be loaded for use. This also depends on the inference server.

Inference *will* be slow. But to start, we want to achieve decentralized p2p inference first. We will then evaluate how much of a bottleneck inference over network is. Details for how to balance traffic/rate-limiting have not been thought of yet and will proceed after the above is working.

## Quick Start

TODO: Startup script to automate this

### Prerequisites

- Go 1.21 or later
- Docker and Docker Compose (optional for now)
- Make (optional, for using Makefile commands)
- Python 3.10 or later
- [protoc](https://protobuf.dev/installation/) and the [go extension for protoc](https://grpc.io/docs/languages/go/quickstart/)

### Local Development

1. Clone and setup dependencies

   ```bash
   go mod download
   ```

2. Run worker node (only functional part currently)

```bash
make run-worker
```

**Outdated**

<del>

1. **Clone and setup dependencies:**

   ```bash
   go mod download
   ```

2. **Setup a local Python environment:**

 ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   pip install -r requirements.txt
   ```

   ```bash
   python3 -m pip install -r backend/python/requirements.txt
   ```

3. **Build proto**
    1. Install protoc and go's [extension](https://grpc.io/docs/languages/go/quickstart/) for protoc

    1. Build protobuf

      ```bash
      # For python spec
      python3 -m grpc_tools.protoc -Ibackend --python_out=backend/python/proto --grpc_python_out=backend/python/proto --pyi_out=backend/python/proto backend/backend.proto
      
      # For go spec
      protoc --go_out=. --go-grpc_out=. backend/backend.proto
      ```

4. Run main to view commands

*Make sure you've activated your Python virtual environment in the same terminal session as you're running the go application.*

   ```bash
   go run main.go
   ```

### Docker Development (WIP)

1. **Build and run with Docker Compose:**

   ```bash
   docker-compose up --build
   # or
   make dev
   ```

2. **Stop the containers:**

   ```bash
   docker-compose down
   # or
   make docker-stop
   ```

## API Endpoints (WIP)

- `GET /health` - Health check endpoint
- `GET /api/v1/hello` - Example API endpoint

## Available Make Commands (WIP)

```bash
make help          # Show available commands
make build         # Build the Go application
make run           # Run the application locally
make test          # Run tests
make docker-build  # Build Docker image
make docker-run    # Run with Docker Compose
make docker-stop   # Stop Docker containers
make dev           # Start development environment
make clean         # Clean build artifacts
```

## Environment Variables

- `PORT` - Server port (default: 8080)
- `GIN_MODE` - Gin mode (debug/release, default: debug)

## Deployment (WIP)

1. **Build the Docker image:**

   ```bash
   make docker-build
   ```

2. **Push to registry (configure DOCKER_REGISTRY in Makefile):**

   ```bash
   make docker-push
   ```

3. **Deploy using Docker Compose:**

   ```bash
   docker-compose up -d
   ```

</del>

## References

This project takes inspiration and code snippets from several sources:

- [LocalAI](https://github.com/mudler/LocalAI)
  - Main source of inspiration. This project aims to take the distributed inference capabilities of LocalAI and extend them to truly decentralized networks that work on more inference servers than just llama.cpp

- [llm-d](https://github.com/llm-d)
  - P/D Dissaggregation, KV Cache aware routing and other distributed systems techniques
 
- [go-libp2p](https://github.com/libp2p/go-libp2p)
  - P2P Capabilities

- [ipfs](https://github.com/ipfs/ipfs)
  - Distributed file systems

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request
