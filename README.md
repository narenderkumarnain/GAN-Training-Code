# GAN-Training-Code
This repository contains modular training code for GAN Experiments.

<hr>

### CycleGAN Implementation 

#### Quick Usage 
#### Imports 
```python
import wandb
import os 
from gan_trainer.cyclegan.model import CycleGAN
from gan_trainer.cyclegan.dataset import ImagetoImageDataset
from torchvision.transforms import transforms
from gan_trainer.cyclegan.config import CycleGANConfig

wandb.login(key = "SSH Key")
```

#### Configuration Options 
```python
config = CycleGANConfig()

# changing various configuration options 
config.n_epochs = 5
config.input_dims = (3,256,256)
config.device = "cuda"
config.dataset_path = dataset_path 
config.checkpoint_dir = checkpoint_path
config.checkpoint_freq = 100 
config.sample_results_dir = os.path.join(checkpoint_path, "sample_res_2") 
config.aligned = True 
config.batch_size = 4
config.n_epochs = 10
config.log_freq = 10 
config.sub_dir_dataset = False 

config.weights_gen_AB = os.path.join(checkpoint_path, "Gen_A_to_B.pth")
config.weights_gen_BA = os.path.join(checkpoint_path, "Gen_B_to_A.pth")
config.weights_dis_A = os.path.join(checkpoint_path, "Dis_A.pth")
config.weights_dis_B = os.path.join(checkpoint_path, "Dis_B.pth")
```

#### Create the Model and Train
```python
model = CycleGAN(config, project_name = "Colab-CycleGAN-2")

model.train()
```




