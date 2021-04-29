# MonoDepthAttacks

### Adversarial attacks on state of the art monocular depth estimation networks
-  Contains FCRN, AdaBins, and DPT depth estimation networks
-  Implements PGD, FGSM, and MI-FGSM adversarial attacks
    -  Non-targeted and targeted versions with L1, L2, and Reverse Huber loss options
-  Support for KITTI and NYUv2 depth datasets

FCRN reference: https://arxiv.org/abs/1606.00373  
AdaBins reference: https://arxiv.org/abs/2011.14141  
DPT reference: https://arxiv.org/abs/2103.13413  

## Usage
This work runs on Python 3 and PyTorch 1.6+

### Installation
Install dependencies
- PyTorch (https://pytorch.org/get-started/locally/)
- Numpy
- Matplotlib 
- PIL
- natsort

Clone this repo:  
```git clone https://github.com/danielzgsilva/MonoDepthAttacks```  
```cd MonoDepthAttacks```

Download the NYUv2 dataset: http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat  
Download the KITTI Raw dataset: http://www.cvlibs.net/datasets/kitti/raw_data.php  
Download the KITTI Depth dataset: http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction  

Configure your dataset paths in "dataloaders/path.py".

### Training
Examples for training an FCRN model from scratch:  
`python main.py --dataset kitti --lr 0.001 --epochs 20 --optim adam --resnet_layers 50 --loss berhu`  
`python main.py --dataset nyu --lr 0.001 --epochs 20 --optim adam --resnet_layers 18 --loss l1`  

Example for finetuning a FCRN model via FGSM adversarial training:  
`python main.py --model resnet --dataset kitti --lr 0.0001 --epochs 10 --optim adam --resnet_layers 50 --loss l2 --resume /path/to/model --adv_training True --attack mifgsm --iterations 1 --epsilon 5 --alpha 5`  

We do not support training AdaBins or DPT from scratch, but pretrained models can be found here:  
`AdaBins: https://github.com/shariqfarooq123/AdaBins`  
`DPT: https://github.com/intel-isl/DPT`  
 
### Evaluation 
Evaluating a given model:  
`python eval.py --dataset kitti --model dpt--resume /path/to/model --attack none`  

Attacking and evaluating a given model:  
`python eval.py --dataset nyu --model adabins --resume /path/to/model --attack pgd --epsilon 3 --iterations 7 --loss l1`  
`python eval.py --dataset kitti --model dpt --resume /path/to/model --attack mifgsm --targeted True  --move_target 1.0`  

(Note that the above are simply examples and do not necessarily result in optimal performance)  

List of available arguments:
- **--num_agents** | number of agents to spawn into the world
- **--model** | model to use (resnet, adabins, dpt)
- **--attack** | attack to run (pgd, mifgsm)
- **--adv_training** | perform adversarial training
- **--eval_output_dir** | directory to save evaluation results and images
- **--decoder** | type of FCRN decoder (upproj, upconv, deconv, fasterupproj)
- **--resnet_layers** | number of layers in FCRN encoder (18, 34, 50, 101, 152)
- **--resume** | path of model to load
- **--batch-size** | mini-batch size
- **--loss** | l1, l2, or berhu
- **--epochs** | help='number of total epochs to run 
- **--optim** | pytorch optimizer for training (sgd, adam)
- **--learning-rate** | initial learning rate
- **--lr_patience** | patience of LR scheduler
- **--scheduler** | learning rate scheduler during training 
- **--momentum** | momentum term for optimizer if applicable
- **--weight_decay** | weight decay for optimizer if applicable
- **--workers** | number of data loading workers
- **--dataset** | nyu or kitti
- **--manual_seed** | manually set random seed
- **--print-freq** | print frequency of metrics during training or eval
- **--targeted** | Choose if adversarial attack is targeted (defaults to attack car class in KITTI)
- **--move_target** | scaling factor by which to perturb depth of targeted class
- **--epsilon** | maximum perturbation magnitude 
- **--iterations** | number of pgd or mi-fgsm iterations
- **--alpha** | step size for pgd or mi-fgsm
- **--g_smooth** | add translational invariance to the adversarial attack
- **--k** | kernel size during guassian smoothing for translation invariance 
