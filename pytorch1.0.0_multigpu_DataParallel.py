import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np 
import argparse, os 
from collections import OrderedDict
#for PIL import Image

# custom weights initialization
def init_weights(layer):
    if isinstance(layer, nn.Linear):
        nn.init.xavier_normal_(layer.weight)
        layer.bias.data.fill_(0.0)
    
    if isinstance(layer, nn.Conv2d):
        nn.init.xavier_normal_(layer.weight)

# optimizer mapping
get_solver = {
    "sgd": torch.optim.SGD,
    "adam": torch.optim.Adam,
}

get_loss_func = {
    "XE": nn.CrossEntropyLoss(),
}

def collate_fn(data):
    # TODO process the data list
    # I do nothing here
    images, labels = zip(*data)
    images = torch.stack(images, 0) # 3D to 4D
    labels = torch.Tensor(labels).long()

    return images, labels

# define model
class LeNet(nn.Module):
    '''
    LeNet-5 model : http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf
    '''
    def __init__(self):

        super(LeNet, self).__init__()
        
        self.layer1 = nn.Sequential(OrderedDict([
                            ('conv1', nn.Conv2d(1, 6, 5, 1)),
                            ('conv1_relu', nn.ReLU()),
                            ('maxpool1', nn.MaxPool2d(2, 2))
                        ]))
        
        self.layer2 = nn.Sequential(OrderedDict([
                            ('conv2', nn.Conv2d(6, 16, 5, 1)),
                            ('conv2_relu', nn.ReLU()),
                            ('maxpool2', nn.MaxPool2d(2, 2))
                        ]))

        self.layer3 = nn.Sequential(OrderedDict([
                            ('fc3', nn.Linear(400, 120)),
                            ('fc3_relu', nn.ReLU())
                        ])) 
        
        self.layer4 = nn.Sequential(OrderedDict([
                            ('fc4', nn.Linear(120, 84)),
                            ('fc4_relu', nn.ReLU())
                        ]))

        self.layer5 = nn.Sequential(OrderedDict([
                            ('fc5', nn.Linear(84, 10)),
                            ('fc5_relu', nn.ReLU())
                        ]))

        self.init_weights()


    def init_weights(self):
        # init weights
        self.layer1.apply(init_weights)
        self.layer2.apply(init_weights)
        self.layer3.apply(init_weights)
        self.layer4.apply(init_weights)
        self.layer5.apply(init_weights)

    def forward(self, x):

        out1 = self.layer1(x)

        out2 = self.layer2(out1)

        out2 = out2.view(-1, 400) # flatten

        out3 = self.layer3(out2)

        out4 = self.layer4(out3)

        out = self.layer5(out4)

        return out

# dataset 
class MNISTDataset(data.Dataset):
    """
    Pesudo MNIST Dataset
    """
    def __init__(self, data_dir):
        # TODO load images and labels here
        self.images = []
        self.labels = []
        #self.length = len(self.images)
        self.length = 100

    # return one training sample
    def __getitem__(self, index):
        # TODO get image and label here
        # image = self.images[index]
        # label = self.labels[index]

        image = np.zeros([1, 32, 32]) # fake data
        image = torch.Tensor(image)
        label = 0 # fake data

        return image, label 

    def __len__(self):
        return self.length

class Trainer():

    def __init__(self, opt, model):
        
        self.use_gpu = torch.cuda.is_available()

        # use multi gpu
        self.model = nn.DataParallel(model).cuda() if self.use_gpu else nn.DataParallel(model) 
        self.params = model.parameters() # require_grads are True by default, means train all parameters

        self.learning_rate = opt.learning_rate
        self.solver = get_solver[opt.solver_name](self.params, lr=self.learning_rate)
        self.loss_func = get_loss_func[opt.criterion]

    def train(self, images, labels):
        if self.use_gpu:
            inputs = images.cuda()
            labels = labels.cuda()

        logits = self.model(inputs)

        self.solver.zero_grad() # clear gradients
        loss = self.loss_func(logits, labels)
        loss.backward() # back propagation
        self.solver.step() # update parameters
        return loss.item() # pytorch 1.0.0 for 0-dim tensor

def save_checkpoint(state_dict, save_dir, model_name):
    torch.save(state_dict, os.path.join(save_dir, model_name))

def main(opt):
    if not os.path.exists(opt.model_dir):
        os.makedirs(opt.model_dir)

    # create data loader 
    dset = MNISTDataset(opt.data_dir)
    train_loader = data.DataLoader(dataset=dset,
                                    batch_size=opt.batch_size,
                                    shuffle=True,
                                    num_workers=opt.workers, # subprocess to load data
                                    pin_memory=False, # large memory available set True
                                    collate_fn=collate_fn,
                                    sampler=None) # for distributed training
    
    # create model
    net = LeNet()
    print("number of params ", len(list(net.parameters())))

    # create trainer
    trainer = Trainer(opt, net)
    trainer.model.train() # set train flag
    Iter = 0
    for epoch in range(opt.max_epoches):

        for batch_data in train_loader:

            loss = trainer.train(*batch_data)
            Iter += 1

            if Iter % opt.echo_iter == 0:
                print("Epoch {} Iter {} Loss {}".format(epoch + 1, Iter, loss))

        # save model for each epoch
        save_checkpoint({
                'epoch': epoch + 1,
                'model': net.state_dict(), # model parameters
                'loss': loss, # loss 
                'opt': opt, # model setting
                'iter': Iter, # training Iter
            }, opt.model_dir, "test_epoch{}_iter{}".format(epoch + 1, Iter))



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data/mnist',
                        help='path to datasets')
    parser.add_argument('--model_dir', default='models/lenet',
                        help='path to save model')
    parser.add_argument('--learning_rate', default=0.001,
                        help='learning rate')
    parser.add_argument('--solver_name', default='sgd',
                        help='solver name')
    parser.add_argument('--criterion', default='XE',
                        help='loss function')
    parser.add_argument('--batch_size', default=64,
                        help='batch size')
    parser.add_argument('--workers', default=2,
                        help='batch size')
    parser.add_argument('--max_epoches', default=50,
                        help='max epoches')
    parser.add_argument('--echo_iter', default=10,
                        help='echo info every iter')
    opt = parser.parse_args()
    print(opt)

    main(opt)



        







        
