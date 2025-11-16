# abstract base class
import torch

class BaseModel():
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def train(self):
        pass

    def evaluate(self, mode='val'):
        pass

    def save_model(self, ckpt):
        pass
    
    def build_model(self, configs):
        pass
    
    def build_dataset(self, configs):
        pass
    
    def build_dataloader(self, dataset, configs):
        pass

    def build_optimizer(self, configs):
        pass
    
    def build_scheduler(self, configs):
        pass
    
    def build_criterion(self, configs):
        pass
    
    def build_metrics(self, configs):
        pass

    def load_model(self, ckpt):
        pass
    
    def get_cosine_schedule_with_warmup(
        self, 
        optimizer, 
        num_warmup_steps, 
        num_training_steps
    ):
        def _lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            no_progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
            return max(0, math.cos(math.pi * num_cycles * no_progress))
        return LambdaLR(optimizer, _lr_lambda, last_epoch=-1)