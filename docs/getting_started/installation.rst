Installation
=============

System Requirements
-------------------

**Hardware:**

- NVIDIA GPU with 11GB+ VRAM (tested on RTX 2080 Ti)
- Supported GPUs: RTX 20 series and newer, A-series, H-series

**Software:**

- Python 3.11+
- NVIDIA Driver 470+
- CUDA Toolkit 11.5+

**Optional:**

- Kaggle account (for model weights)
- HuggingFace token (for model upload)

CUDA and Driver Setup
---------------------

Check your current setup::

    nvidia-smi           # Shows driver and CUDA version
    nvcc --version       # Shows CUDA Toolkit version

If you need to install or update CUDA, visit: https://developer.nvidia.com/cuda-toolkit

For driver installation, visit: https://www.nvidia.com/Download/driverDetails.aspx

Installation Methods
--------------------

From Source
^^^^^^^^^^^

Clone the repository and install in development mode::

    git clone https://github.com/yourusername/agent-tunix.git
    cd agent-tunix
    pip install -e .

With Development Tools
^^^^^^^^^^^^^^^^^^^^^^

For development and testing::

    pip install -e ".[dev]"

This installs additional dependencies for:

- Testing (pytest, pytest-cov)
- Code formatting (black, ruff)
- Type checking (mypy)
- Documentation (sphinx, sphinx-rtd-theme)

Verify Installation
-------------------

Check that everything is properly installed::

    # Verify Python version
    python --version

    # Verify GPU access
    python -c "import jax; print(jax.devices())"

    # Verify package installation
    python -c "import agent_tunix; print(agent_tunix.__version__)"

    # Or use the Makefile
    make check-gpu
    make show-config

Environment Variables
---------------------

Optional environment variables for configuration:

**HuggingFace Token** (for model uploads)::

    export HF_TOKEN=your_token_here

**Weights and Biases** (for experiment tracking)::

    export WANDB_PROJECT=your_project_name

**CUDA Configuration** (if needed)::

    export CUDA_HOME=/usr/local/cuda-13.0
    export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH

Create a `.env` file in the project root to automatically load these::

    cat > .env << EOF
    HF_TOKEN=your_token
    WANDB_PROJECT=your_project
    EOF

Troubleshooting
---------------

**CUDA not detected**

If JAX can't find your GPU::

    python -c "import jax; print(jax.devices())"

Set CUDA paths and retry::

    export CUDA_HOME=/path/to/cuda
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

**Out of Memory (OOM)**

Reduce batch size or model size in configuration::

    python run_training.py training.micro_batch_size=1 model=gemma3_270m

**Kaggle Authentication**

For model weights, authenticate with Kaggle::

    kaggle auth login

See `Kaggle API Documentation <https://github.com/Kaggle/kaggle-api>`_ for details.

Next Steps
----------

- :doc:`Quick Start </getting_started/quick_start>`
- :doc:`Configuration Guide </getting_started/configuration>`
- :doc:`Training Guide </guide/training>`
