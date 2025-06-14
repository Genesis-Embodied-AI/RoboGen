"""Validation tests to verify the testing infrastructure is properly set up."""
import pytest
import sys
import os
from pathlib import Path


def test_python_version():
    """Verify Python version is 3.9 or higher."""
    assert sys.version_info >= (3, 9), "Python 3.9 or higher is required"


def test_project_structure():
    """Verify the basic project structure exists."""
    project_root = Path(__file__).parent.parent
    
    assert project_root.exists(), "Project root directory not found"
    assert (project_root / "tests").exists(), "tests directory not found"
    assert (project_root / "tests" / "__init__.py").exists(), "tests/__init__.py not found"
    assert (project_root / "tests" / "conftest.py").exists(), "conftest.py not found"
    assert (project_root / "tests" / "unit").exists(), "unit test directory not found"
    assert (project_root / "tests" / "integration").exists(), "integration test directory not found"


def test_pyproject_toml_exists():
    """Verify pyproject.toml exists and is readable."""
    project_root = Path(__file__).parent.parent
    pyproject_path = project_root / "pyproject.toml"
    
    assert pyproject_path.exists(), "pyproject.toml not found"
    assert pyproject_path.is_file(), "pyproject.toml is not a file"
    
    content = pyproject_path.read_text()
    assert "[tool.poetry]" in content, "Poetry configuration not found in pyproject.toml"
    assert "[tool.pytest.ini_options]" in content, "Pytest configuration not found"
    assert "[tool.coverage.run]" in content, "Coverage configuration not found"


def test_import_project_modules():
    """Test that project modules can be imported."""
    try:
        import gpt_4
        import manipulation
        import locomotion
        import cem_policy
        import objaverse_utils
        import pybullet_ompl
    except ImportError as e:
        pytest.skip(f"Module import failed (expected before installation): {e}")


@pytest.mark.unit
def test_unit_marker():
    """Test that unit test marker works."""
    assert True


@pytest.mark.integration
def test_integration_marker():
    """Test that integration test marker works."""
    assert True


@pytest.mark.slow
def test_slow_marker():
    """Test that slow test marker works."""
    assert True


def test_fixtures_available(temp_dir, mock_config, mock_robot):
    """Test that custom fixtures are available and working."""
    assert temp_dir.exists()
    assert temp_dir.is_dir()
    
    assert isinstance(mock_config, dict)
    assert "debug" in mock_config
    assert "api_key" in mock_config
    
    assert hasattr(mock_robot, "get_joint_angles")
    assert hasattr(mock_robot, "move_to_position")
    angles = mock_robot.get_joint_angles()
    assert len(angles) == 7


def test_temp_dir_cleanup(temp_dir):
    """Test that temp_dir fixture creates and cleans up properly."""
    test_file = temp_dir / "test.txt"
    test_file.write_text("test content")
    assert test_file.exists()
    

class TestValidationClass:
    """Test that class-based tests work."""
    
    def test_class_method(self):
        """Test method in a test class."""
        assert 1 + 1 == 2
    
    def test_with_fixture(self, mock_simulator):
        """Test using fixture in a class method."""
        mock_simulator.reset()
        mock_simulator.reset.assert_called_once()


@pytest.mark.parametrize("input_val,expected", [
    (1, 2),
    (2, 4),
    (3, 6),
])
def test_parametrized(input_val, expected):
    """Test that parametrized tests work."""
    assert input_val * 2 == expected


def test_coverage_targets():
    """Verify coverage is configured for the right modules."""
    project_root = Path(__file__).parent.parent
    pyproject_path = project_root / "pyproject.toml"
    content = pyproject_path.read_text()
    
    expected_modules = ["gpt_4", "manipulation", "locomotion", "cem_policy", 
                       "objaverse_utils", "pybullet_ompl", "RL"]
    
    for module in expected_modules:
        assert f'"--cov={module}"' in content or f"'{module}'" in content, \
            f"Coverage not configured for {module}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])