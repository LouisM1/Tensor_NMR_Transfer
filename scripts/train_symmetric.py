"""Train the symmetric baseline using the inherited graph-pes compatibility fixes."""
from pathlib import Path
import argparse,os,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.chdir(ROOT)
import yaml
from nmr_helpers.shared_head_training import train_from_config
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('config',nargs='?',default='configs/tensor/01-SymmetricTensor-Scratch-N100.yaml')
args=parser.parse_args()
train_from_config(yaml.safe_load(Path(args.config).read_text()))
