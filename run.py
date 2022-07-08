import pytorch_lightning as pl
import hydra
from omegaconf import DictConfig

@hydra.main(config_path='conf', config_name='config')
def main(cfg: DictConfig):
    print(cfg)
    

if __name__ == "__main__":
    main()