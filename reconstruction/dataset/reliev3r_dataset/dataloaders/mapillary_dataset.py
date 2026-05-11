from .base_dataset import Realistic_Dataset


class Mapillary_Dataset(Realistic_Dataset):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
