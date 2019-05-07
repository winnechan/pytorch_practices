# pytorch practices
notes of pytorch practices in experiments

### Requirements
PyTorch 1.0.0 or PyTorch 0.4.0

### Content
1. [single PC with multiple GPUs using DataParallel](https://github.com/winnechan/effective_pytorch/blob/master/pytorch1.0.0_multigpu_DataParallel.py) (DataParallel may hang with PyTorch 0.4.0, Tesla V100/K80 due to nccl issue)

2. [single PC with multiple GPUs using DistributedDataParallel](https://github.com/winnechan/pytorch_practices/blob/master/pytorch1.0.0_multigpu_DistributedDataParallel.py)

3. [specify different learning rates for different layers](https://github.com/winnechan/pytorch_practices/blob/master/specify_different_lr_for_different_layers.png)

4. [load model trained on multi gpus using torch.save({"model": model.state_dict()}, "xxx") to save instance of DataParallel](https://github.com/winnechan/pytorch_practices/blob/master/loading_models_trained_on_multigpus.png):
build a new OrderedDict with keys removing "module"

5. [save model trained on multi gpus in order to load it without multi gpus](https://github.com/winnechan/pytorch_practices/blob/master/saveing_models_trained_on_multigpus.png):
save the model without DataParallel wrap

6. load model trained on multi gpus by wrapping it in DataParallel again

7. [word embedding tutorial](https://github.com/winnechan/pytorch_practices/blob/master/word_embeddings_tutorial.ipynb)

8. [sequence models tutorial](https://github.com/winnechan/pytorch_practices/blob/master/sequence_models_tutorial.ipynb)
