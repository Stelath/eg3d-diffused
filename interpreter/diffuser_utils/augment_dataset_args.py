import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Dataset Augmentation script for EG3D Diffuser Model.")
    parser.add_argument(
        "--dataset",
        type=str,
        default='data/',
        help="Existing dataset location.",
    )
    parser.add_argument("--model-path", type=str, help="Location of the model if needed")
    parser.add_argument("--gpu", type=bool, default=True, help="Whether to use the GPU or not")
    parser.add_argument("--faces", type=bool, default=False)
    parser.add_argument("--ae", type=bool, default=False)
    
    args = parser.parse_args()
    
    # if not os.path.exists(args.model_path):
    #     raise Exception("Model path does not exist.")
    
    if not os.path.exists(args.dataset):
        raise Exception("Dataset doesn't exist")
    
    imgs_path = os.path.join(args.dataset, 'imgs')
    if not os.path.exists(imgs_path):
        raise Exception("imgs/ directory doesn't exist")
    
    df_path = os.path.join(args.dataset, 'dataset.df')
    if not os.path.exists(df_path):
        raise Exception("dataset.df doesn't exist")
    
    return args
