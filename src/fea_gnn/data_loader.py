from data.data_pipeline import PlateHoleDataset


def get_dataset(root="data/"):
    """Charge le dataset de plaques trouées généré précédemment."""
    return PlateHoleDataset(root=root)
