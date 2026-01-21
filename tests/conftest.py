import os

import pytest
import yaml


@pytest.fixture(scope="session", autouse=True)
def create_test_config():
    """
    Crée un config.yaml de test à la racine AVANT que les tests ne commencent.
    scope='session' garantit qu'il est créé une seule fois pour tous les tests.
    autouse=True assure que la fixture est exécutée sans appel explicite.
    """
    config_data = {
        "geometry": {"nx": 5, "ny": 2, "length": 1.0, "height": 0.2},
        "material": {
            "youngs_modulus_range": [200.0, 210.0],
            "poissons_ratio_range": [0.3, 0.3],
        },
        "model": {"input_dim": 5, "hidden_dim": 16, "layers": 2},
        "training": {
            "learning_rate": 0.001,
            "epochs": 1,
            "batch_size": 2,
            "physics_weight": 0.01,
        },
        "env": {"device": "cpu", "save_path": "./models"},
    }

    # On force l'écriture à la racine du projet pour que les imports le trouvent
    config_path = os.path.join(os.getcwd(), "config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config_data, f)

    yield

    # Nettoyage
    if os.path.exists(config_path):
        os.remove(config_path)
