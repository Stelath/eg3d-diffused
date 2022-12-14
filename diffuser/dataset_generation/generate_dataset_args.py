import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Dataset Generation script for EG3D Diffuser Model.")
    parser.add_argument(
        "--model_path",
        type=str,
        default='eg3d_model/ffhqrebalanced512-128.pkl',
        help=(
            "The path to the model to use for generating images, defaults to:",
            "(eg3d_model/ffhqrebalanced512-128.pkl)",
        ),
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1000,
        help="The amount of images to generate for the dataset, defaults to 1000.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default='../data/',
        help="Where to output the generated datset.",
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path):
        raise Exception("Model path does not exist.")
    
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    
    imgs_path = os.path.join(args.out_dir, 'imgs')
    if not os.path.exists(imgs_path):
        os.mkdir(imgs_path)
    
    return args
