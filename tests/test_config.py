"""Fast tests for configuration loading and composition."""

import pytest
from hydra import compose, initialize_config_dir
from pathlib import Path
from omegaconf import DictConfig, OmegaConf


@pytest.fixture
def config_dir():
    """Get path to config directory."""
    return str(Path(__file__).parent.parent / "conf")


class TestConfigLoading:
    """Test configuration loading."""

    def test_default_config_loads(self, config_dir):
        """Test that default configuration loads without errors."""
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            cfg = compose(config_name="config")
            assert cfg is not None
            assert isinstance(cfg, DictConfig)

    def test_config_has_required_sections(self, config_dir):
        """Test that configuration has all required sections."""
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            cfg = compose(config_name="config")

            required_sections = [
                "model", "optimizer", "scheduler", "grpo",
                "generation", "training"
            ]

            for section in required_sections:
                assert section in cfg, f"Missing section: {section}"

    def test_model_config_valid(self, config_dir):
        """Test that model configuration is valid."""
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            cfg = compose(config_name="config")

            model = cfg.model
            assert model.model_family == "gemma3"
            assert model.model_size in ["270m", "1b", "4b"]
            assert model.lora_rank > 0
            assert model.lora_alpha > 0

    def test_optimizer_config_valid(self, config_dir):
        """Test that optimizer configuration is valid."""
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            cfg = compose(config_name="config")

            optimizer = cfg.optimizer
            assert optimizer.learning_rate > 0
            assert 0 <= optimizer.warmup_ratio <= 1
            assert optimizer.max_grad_norm > 0
            assert optimizer.beta1 < optimizer.beta2

    def test_training_config_valid(self, config_dir):
        """Test that training configuration is valid."""
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            cfg = compose(config_name="config")

            training = cfg.training
            assert training.micro_batch_size > 0
            assert training.num_batches > 0
            assert training.num_epochs > 0
            assert training.num_test_batches > 0

    def test_grpo_config_valid(self, config_dir):
        """Test that GRPO configuration is valid."""
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            cfg = compose(config_name="config")

            grpo = cfg.grpo
            assert grpo.num_generations > 0
            assert grpo.beta > 0
            assert 0 < grpo.epsilon < 1

    def test_generation_config_valid(self, config_dir):
        """Test that generation configuration is valid."""
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            cfg = compose(config_name="config")

            gen = cfg.generation
            assert gen.max_prompt_length > 0
            assert gen.max_generation_steps > 0
            assert 0 < gen.temperature
            assert 0 <= gen.top_p <= 1
            assert gen.top_k > 0


class TestExperimentConfigs:
    """Test experiment preset configurations."""

    def test_quick_test_experiment(self, config_dir):
        """Test quick_test experiment configuration."""
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            cfg = compose(config_name="config", overrides=["+experiment=quick_test"])

            assert cfg.experiment_name == "quick_test"
            assert "quick" in cfg.tags
            assert cfg.training.num_batches == 10
            assert cfg.training.micro_batch_size == 1
            assert cfg.model.lora_rank == 8
            assert cfg.grpo.num_generations == 2
            assert cfg.wandb_disabled is True
            assert cfg.debug is True

    def test_full_training_experiment(self, config_dir):
        """Test full_training experiment configuration."""
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            cfg = compose(config_name="config", overrides=["+experiment=full_training"])

            assert cfg.experiment_name == "full_training"
            assert "production" in cfg.tags
            assert cfg.training.num_batches == 3738
            assert cfg.training.micro_batch_size == 4
            assert cfg.model.lora_rank == 32
            assert cfg.grpo.num_generations == 4
            assert cfg.wandb_disabled is False
            assert cfg.debug is False


class TestConfigOverrides:
    """Test configuration parameter overrides."""

    def test_learning_rate_override(self, config_dir):
        """Test overriding learning rate."""
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            cfg = compose(
                config_name="config",
                overrides=["optimizer.learning_rate=1e-4"]
            )
            assert cfg.optimizer.learning_rate == 1e-4

    def test_batch_size_override(self, config_dir):
        """Test overriding batch size."""
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            cfg = compose(
                config_name="config",
                overrides=["training.micro_batch_size=8"]
            )
            assert cfg.training.micro_batch_size == 8

    def test_lora_rank_override(self, config_dir):
        """Test overriding LoRA rank."""
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            cfg = compose(
                config_name="config",
                overrides=["model.lora_rank=64"]
            )
            assert cfg.model.lora_rank == 64

    def test_multiple_overrides(self, config_dir):
        """Test multiple simultaneous overrides."""
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            cfg = compose(
                config_name="config",
                overrides=[
                    "optimizer.learning_rate=5e-6",
                    "training.num_batches=100",
                    "model.lora_rank=16"
                ]
            )
            assert cfg.optimizer.learning_rate == 5e-6
            assert cfg.training.num_batches == 100
            assert cfg.model.lora_rank == 16


class TestConfigConversion:
    """Test configuration conversion and serialization."""

    def test_config_to_dict(self, config_dir):
        """Test converting config to dictionary."""
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            cfg = compose(config_name="config")
            cfg_dict = OmegaConf.to_container(cfg, resolve=True)

            assert isinstance(cfg_dict, dict)
            assert "model" in cfg_dict
            assert "optimizer" in cfg_dict

    def test_config_to_yaml(self, config_dir):
        """Test converting config to YAML string."""
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            cfg = compose(config_name="config")
            yaml_str = OmegaConf.to_yaml(cfg)

            assert isinstance(yaml_str, str)
            assert "model:" in yaml_str
            assert "optimizer:" in yaml_str
            assert "training:" in yaml_str
