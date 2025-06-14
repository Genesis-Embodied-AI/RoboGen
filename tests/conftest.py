"""Shared pytest fixtures and configuration."""
import os
import sys
import tempfile
import shutil
from pathlib import Path
from typing import Generator, Dict, Any
import pytest
from unittest.mock import Mock, MagicMock


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def mock_config() -> Dict[str, Any]:
    """Provide a mock configuration dictionary."""
    return {
        "debug": False,
        "verbose": True,
        "max_retries": 3,
        "timeout": 30,
        "api_key": "test-api-key",
        "model": "test-model",
        "temperature": 0.7,
    }


@pytest.fixture
def mock_robot():
    """Create a mock robot object for testing."""
    robot = MagicMock()
    robot.get_joint_angles.return_value = [0.0] * 7
    robot.get_end_effector_position.return_value = [0.5, 0.0, 0.5]
    robot.get_end_effector_orientation.return_value = [0.0, 0.0, 0.0, 1.0]
    robot.move_to_position.return_value = True
    robot.grasp.return_value = True
    robot.release.return_value = True
    return robot


@pytest.fixture
def mock_simulator():
    """Create a mock simulator object."""
    sim = MagicMock()
    sim.step.return_value = None
    sim.reset.return_value = None
    sim.get_object_position.return_value = [0.0, 0.0, 0.0]
    sim.get_object_orientation.return_value = [0.0, 0.0, 0.0, 1.0]
    sim.spawn_object.return_value = 1
    sim.remove_object.return_value = True
    return sim


@pytest.fixture
def mock_api_client():
    """Create a mock API client for testing GPT interactions."""
    client = MagicMock()
    client.query.return_value = {
        "choices": [{
            "message": {
                "content": "Test response"
            }
        }]
    }
    client.verify.return_value = True
    return client


@pytest.fixture
def sample_task_config() -> Dict[str, Any]:
    """Provide a sample task configuration."""
    return {
        "task_name": "test_task",
        "description": "A test task for unit testing",
        "steps": [
            {"action": "grasp", "object": "test_object"},
            {"action": "move", "position": [1.0, 0.0, 0.5]},
            {"action": "release"}
        ],
        "success_criteria": {
            "object_position": [1.0, 0.0, 0.5],
            "tolerance": 0.1
        }
    }


@pytest.fixture
def mock_motion_planner():
    """Create a mock motion planner."""
    planner = MagicMock()
    planner.plan.return_value = [[0.0] * 7, [0.1] * 7, [0.2] * 7]
    planner.is_valid_configuration.return_value = True
    planner.check_collision.return_value = False
    return planner


@pytest.fixture
def mock_environment():
    """Create a mock environment for testing."""
    env = MagicMock()
    env.reset.return_value = {"observation": [0.0] * 10}
    env.step.return_value = (
        {"observation": [0.1] * 10},  # next_obs
        0.5,  # reward
        False,  # done
        {"success": False}  # info
    )
    env.render.return_value = None
    return env


@pytest.fixture
def sample_urdf_path(temp_dir: Path) -> Path:
    """Create a sample URDF file for testing."""
    urdf_content = """<?xml version="1.0"?>
<robot name="test_robot">
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.1 0.1 0.1"/>
      </geometry>
    </visual>
  </link>
</robot>"""
    urdf_path = temp_dir / "test_robot.urdf"
    urdf_path.write_text(urdf_content)
    return urdf_path


@pytest.fixture
def mock_logger():
    """Create a mock logger for testing."""
    logger = MagicMock()
    logger.info = Mock()
    logger.debug = Mock()
    logger.warning = Mock()
    logger.error = Mock()
    logger.critical = Mock()
    return logger


@pytest.fixture(autouse=True)
def reset_environment():
    """Reset environment variables before each test."""
    original_env = os.environ.copy()
    yield
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def capture_stdout(monkeypatch):
    """Capture stdout for testing print statements."""
    import io
    captured_output = io.StringIO()
    monkeypatch.setattr('sys.stdout', captured_output)
    yield captured_output
    

@pytest.fixture
def mock_ray():
    """Mock Ray for distributed computing tests."""
    ray_mock = MagicMock()
    ray_mock.init.return_value = None
    ray_mock.shutdown.return_value = None
    ray_mock.get.return_value = {"result": "success"}
    ray_mock.put.return_value = "object_id_123"
    return ray_mock


def pytest_configure(config):
    """Configure pytest with custom settings."""
    config.addinivalue_line(
        "markers", "requires_gpu: mark test as requiring GPU"
    )
    config.addinivalue_line(
        "markers", "requires_internet: mark test as requiring internet connection"
    )
    config.addinivalue_line(
        "markers", "expensive: mark test as computationally expensive"
    )