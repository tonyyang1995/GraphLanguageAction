from torch_geometric.data import InMemoryDataset
import os.path as osp 
import torch
import pandas as pd

class PygPredictionMoleculeDataset(InMemoryDataset):
    def __init__(self, 
        configs: dict,
    ):
        self.dataset_name = configs["dataset"]["name"]
        self.root = osp.join(configs["dataset"]["path"], self.dataset_name)
        super(PygPredictionMoleculeDataset, self).__init__(self.root, None, None)

        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
        
        self.num_tasks = configs["dataset"]["num_tasks"]
        self.start_column = configs["dataset"]["start_column"]

        self.process_paths = osp.join(self.root, "processed", "geometric_data_processed.pt")

    def get_idx_split(self):
        path = osp.join(self.root, 'split', 'scaffold', 'split_dict.pt')
        if osp.exists(path):
            return torch.load(path)
        else:
            raise ValueError("Split not found, please run data preparation script first")
    @property
    def raw_file_names(self):
        return ["assays.csv.gz"]
    
    @property
    def processed_file_names(self):
        return ["geometric_data_processed.pt"]
    
    def download(self):
        raise ValueError("Download not supported for this dataset")
    
    def process(self):
        raise ValueError("Process not supported for this dataset")
    
    def __repr__(self):
        return "{}()".format(self.__class__.__name__)

