# MonoDepthAttacks

### Adversarial attacks on state of the art monocular depth estimation networks
-  Contains FCRN, AdaBins, and DPT depth estimation networks
-  Implements PGD, FGSM, and MI-FGSM adversarial attacks with L1, L2, and Reverse Huber loss
-  Support for KITTI and NYUv2 depth datasets

FCRN reference: https://arxiv.org/abs/1606.00373
AdaBins reference: https://arxiv.org/abs/2011.14141
DPT reference: https://arxiv.org/abs/2103.13413

## Usage
This work runs on Python 3 and PyTorch 1.6+




`python run.py --num_agents 2 --starts 0,0 5,5 --goals 49,49 45,45 --max_time 30`

`python run.py --num_agents 2 --starts 0,0 5,5 --goals 49,49 45,45 --max_time 30`

List of available commands:
- **--num_agents** | number of agents to spawn into the world 
