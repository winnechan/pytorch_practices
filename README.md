# effective_pytorch
notes of pytorch practices in experiments

### Requirements
PyTorch 1.0.0 or PyTorch 0.4.0

### Content
1. [single PC with multiple GPUs using DataParallel](https://github.com/winnechan/effective_pytorch/blob/master/pytorch1.0.0_multigpu_DataParallel.py) (DataParallel may hang with PyTorch 0.4.0, Tesla V100/K80 due to nccl issue)

2. single PC with multiple GPUs using DistributedDataParallel
