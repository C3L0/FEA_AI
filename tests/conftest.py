import pytest
import yaml
import os


@pytest.fixture(autocmd=True)
def create_test_config():
    """
    This fixture runs automatically before every test.
    It creates a dummy config.yaml in the root folder so the code doesn't crash.
    """
    config_data = {
        "geometry": {"nx": 10, "ny": 2, "length": 1.0, "height": 0.1},
        "material": {
            "youngs_modulus_range": [100.0, 210.0],
            "poissons_ratio_range": [0.2, 0.4],
        },
        "model": {"input_dim": 5, "hidden_dim": 16, "layers": 2},
        "training": {"learning_rate": 0.001, "epochs": 1, "batch_size": 2},
        "env": {"device": "cpu", "save_path": "./models"},
    }

    with open("config.yaml", "w") as f:
        yaml.dump(config_data, f)

    yield  # The test runs here

    # Cleanup after test if needed (optional)
    if os.path.exists("config.yaml"):
        os.remove("config.yaml")
