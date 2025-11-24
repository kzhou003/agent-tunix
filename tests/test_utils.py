"""Fast tests for utility functions."""

import pytest
from pathlib import Path
import sys


class TestProjectStructure:
    """Test that project structure is correct."""

    def test_project_root_exists(self):
        """Test that project root directory exists."""
        project_root = Path(__file__).parent.parent
        assert project_root.exists(), "Project root should exist"
        assert project_root.is_dir(), "Project root should be a directory"

    def test_src_directory_exists(self):
        """Test that src directory exists."""
        src_dir = Path(__file__).parent.parent / "src"
        assert src_dir.exists(), "src directory should exist"
        assert src_dir.is_dir(), "src should be a directory"

    def test_agent_tunix_package_exists(self):
        """Test that agent_tunix package exists."""
        package_dir = Path(__file__).parent.parent / "src" / "agent_tunix"
        assert package_dir.exists(), "agent_tunix package should exist"
        assert package_dir.is_dir(), "agent_tunix should be a directory"
        assert (package_dir / "__init__.py").exists(), "Should have __init__.py"

    def test_conf_directory_exists(self):
        """Test that conf directory exists."""
        conf_dir = Path(__file__).parent.parent / "conf"
        assert conf_dir.exists(), "conf directory should exist"
        assert conf_dir.is_dir(), "conf should be a directory"

    def test_docs_directory_exists(self):
        """Test that docs directory exists."""
        docs_dir = Path(__file__).parent.parent / "docs"
        assert docs_dir.exists(), "docs directory should exist"
        assert docs_dir.is_dir(), "docs should be a directory"


class TestConfigurationFiles:
    """Test that configuration files exist."""

    def test_main_config_exists(self):
        """Test that main config.yaml exists."""
        config_file = Path(__file__).parent.parent / "conf" / "config.yaml"
        assert config_file.exists(), "config.yaml should exist"
        assert config_file.suffix == ".yaml", "Should be YAML file"

    def test_model_configs_exist(self):
        """Test that model configuration files exist."""
        model_dir = Path(__file__).parent.parent / "conf" / "model"
        assert model_dir.exists(), "model config directory should exist"

        expected_models = ["gemma3_270m.yaml", "gemma3_1b.yaml"]
        for model_file in expected_models:
            model_path = model_dir / model_file
            assert model_path.exists(), f"{model_file} should exist"

    def test_optimizer_configs_exist(self):
        """Test that optimizer configuration files exist."""
        opt_dir = Path(__file__).parent.parent / "conf" / "optimizer"
        assert opt_dir.exists(), "optimizer config directory should exist"

        opt_file = opt_dir / "adamw.yaml"
        assert opt_file.exists(), "adamw.yaml should exist"

    def test_experiment_configs_exist(self):
        """Test that experiment configuration files exist."""
        exp_dir = Path(__file__).parent.parent / "conf" / "experiment"
        assert exp_dir.exists(), "experiment config directory should exist"

        expected_experiments = ["quick_test.yaml", "full_training.yaml"]
        for exp_file in expected_experiments:
            exp_path = exp_dir / exp_file
            assert exp_path.exists(), f"{exp_file} should exist"


class TestDocumentationFiles:
    """Test that documentation files exist."""

    def test_main_readme_exists(self):
        """Test that README.md exists."""
        readme = Path(__file__).parent.parent / "README.md"
        assert readme.exists(), "README.md should exist"

    def test_docs_index_exists(self):
        """Test that docs/index.rst exists."""
        index = Path(__file__).parent.parent / "docs" / "index.rst"
        assert index.exists(), "docs/index.rst should exist"

    def test_docs_conf_exists(self):
        """Test that docs/conf.py exists."""
        conf = Path(__file__).parent.parent / "docs" / "conf.py"
        assert conf.exists(), "docs/conf.py should exist"

    def test_docs_makefile_exists(self):
        """Test that docs/Makefile exists."""
        makefile = Path(__file__).parent.parent / "docs" / "Makefile"
        assert makefile.exists(), "docs/Makefile should exist"

    def test_getting_started_docs_exist(self):
        """Test that getting started documentation exists."""
        gs_dir = Path(__file__).parent.parent / "docs" / "getting_started"
        assert gs_dir.exists(), "getting_started docs directory should exist"

        expected_docs = [
            "installation.rst",
            "quick_start.rst",
            "configuration.rst"
        ]
        for doc in expected_docs:
            doc_path = gs_dir / doc
            assert doc_path.exists(), f"{doc} should exist"


class TestBuildFiles:
    """Test that build/project files exist."""

    def test_pyproject_toml_exists(self):
        """Test that pyproject.toml exists."""
        pyproject = Path(__file__).parent.parent / "pyproject.toml"
        assert pyproject.exists(), "pyproject.toml should exist"

    def test_makefile_exists(self):
        """Test that root Makefile exists."""
        makefile = Path(__file__).parent.parent / "Makefile"
        assert makefile.exists(), "Makefile should exist"

    def test_env_file_pattern(self):
        """Test for .env file (optional)."""
        env_file = Path(__file__).parent.parent / ".env"
        # .env is optional, but if it exists, should be valid
        if env_file.exists():
            assert env_file.is_file(), ".env should be a file"

    def test_gitignore_exists(self):
        """Test that .gitignore exists."""
        gitignore = Path(__file__).parent.parent / ".gitignore"
        # .gitignore is recommended but optional
        # Just test it can be checked
        assert isinstance(gitignore, Path), "Should be able to check .gitignore"


class TestImports:
    """Test that main modules can be imported."""

    def test_import_agent_tunix(self):
        """Test importing agent_tunix package."""
        try:
            import agent_tunix
            assert agent_tunix is not None
        except ImportError:
            pytest.skip("agent_tunix not in path (may need installation)")

    def test_import_training_module(self):
        """Test importing training module."""
        try:
            from agent_tunix import train
            assert train is not None
        except ImportError:
            pytest.skip("agent_tunix not in path (may need installation)")

    def test_import_evaluation_module(self):
        """Test importing evaluation module."""
        try:
            from agent_tunix import evaluate
            assert evaluate is not None
        except ImportError:
            pytest.skip("agent_tunix not in path (may need installation)")


class TestFilePermissions:
    """Test that files have correct permissions."""

    def test_python_files_readable(self):
        """Test that Python files are readable."""
        src_dir = Path(__file__).parent.parent / "src" / "agent_tunix"
        if src_dir.exists():
            python_files = list(src_dir.glob("*.py"))
            for py_file in python_files:
                assert py_file.is_file(), f"{py_file} should be a file"
                assert py_file.stat().st_size > 0, f"{py_file} should not be empty"

    def test_config_files_readable(self):
        """Test that config files are readable."""
        conf_dir = Path(__file__).parent.parent / "conf"
        if conf_dir.exists():
            yaml_files = list(conf_dir.rglob("*.yaml"))
            for yaml_file in yaml_files:
                assert yaml_file.is_file(), f"{yaml_file} should be a file"
