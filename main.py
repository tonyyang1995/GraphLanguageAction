import yaml
import argparse 
import random
import numpy as np
import torch

def train(opt):
    if opt.model == 'infoalign':
        from train_infoalign import InfoAlign
        model = InfoAlign()

    else:
        raise ValueError(f"Invalid model: {opt.model}")

    model.train()
    val_metrics = model.evaluate(mode='val')
    test_metrics = model.evaluate(mode='test')
    print(f"Validation Metrics: {val_metrics[model.configs['dataset']['eval_metric']]:.4f}")
    print(f"Test Metrics: {test_metrics[model.configs['dataset']['eval_metric']]:.4f}")

def val(opt):
    if opt.model == 'infoalign':
        from train_infoalign import InfoAlign
        model = InfoAlign()
    else:
        raise ValueError(f"Invalid model: {opt.model}")

    val_metrics = model.evaluate(mode='val', load_model=True)

    print(f"Validation Metrics: {val_metrics[model.configs['dataset']['eval_metric']]:.4f}")

def test(opt):
    if opt.model == 'infoalign':
        from train_infoalign import InfoAlign
        model = InfoAlign()
    else:
        raise ValueError(f"Invalid model: {opt.model}")

    test_metrics = model.evaluate(mode='test', load_model=True)
    print(f"Test Metrics: {test_metrics[model.configs['dataset']['eval_metric']]:.4f}")

def main(opt):
    if opt.mode == 'train':
        train(opt)
    elif opt.mode == 'val':
        val(opt)
    elif opt.mode == 'test':
        test(opt)
    else:
        raise ValueError(f"Invalid mode: {opt.mode}")

if __name__ == "__main__":

    # fix the random seed for reproducibility
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default='infoalign')
    parser.add_argument("--mode", type=str, default='train', choices=['train','val', 'test'])

    opt = parser.parse_args()
    main(opt)
