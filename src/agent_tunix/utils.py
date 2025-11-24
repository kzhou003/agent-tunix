"""Utility functions for agent-tunix."""

import logging
import subprocess
import sys

log = logging.getLogger(__name__)


def check_gpu():
    """Check GPU availability and memory."""
    try:
        import jax

        devices = jax.devices()
        backend = jax.default_backend()
        print(f"JAX devices: {devices}")
        print(f"Backend: {backend}")
    except Exception as e:
        print(f"JAX not available: {e}")

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            print("\nGPU Status:")
            print(result.stdout)
    except Exception as e:
        print(f"nvidia-smi not available: {e}")


def show_config():
    """Show default training configuration from YAML."""
    from omegaconf import OmegaConf
    from hydra import compose, initialize_config_dir
    from pathlib import Path

    # Get the config directory
    config_dir = Path(__file__).parent.parent.parent / "conf"

    try:
        with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
            cfg = compose(config_name="config")

        print("=" * 60)
        print("Default Training Configuration")
        print("=" * 60)
        print(OmegaConf.to_yaml(cfg))
        print("=" * 60)
    except Exception as e:
        log.error(f"Error loading configuration: {e}")
        sys.exit(1)


def main():
    """Command dispatcher for utility commands."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m agent_tunix.utils <command>")
        print("\nAvailable commands:")
        print("  check-gpu    Check GPU availability")
        print("  show-config  Show default configuration")
        sys.exit(1)

    command = sys.argv[1]

    if command == "check-gpu":
        check_gpu()
    elif command == "show-config":
        show_config()
    else:
        print(f"Unknown command: {command}")
        print("\nAvailable commands: check-gpu, show-config")
        sys.exit(1)


if __name__ == "__main__":
    main()
