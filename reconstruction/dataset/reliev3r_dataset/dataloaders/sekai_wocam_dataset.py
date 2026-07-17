from .base_dataset import Realistic_Dataset


class Sekai_woCam_Dataset(Realistic_Dataset):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
