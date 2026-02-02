from data.data_pipeline import PlateHoleDataset


# Maybe move that to utils
def get_dataset(root="data/"):
    """Load the dataset"""
    return PlateHoleDataset(root=root)
