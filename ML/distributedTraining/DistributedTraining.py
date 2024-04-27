from SetupInfo import SlurmSetup

from dataclasses import dataclass
from torch import Tensor
from torch.nn import L1Loss, Linear, Module, Sequential
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from datetime import  datetime

from data import get_data_loaders
from optimization import get_optimizer, get_loss
from train import optimize, one_epoch_test
from transfer import get_model_transfer_learning
from transfer_models import get_model_transfer_learning as get_models
from model import Model

import torch


batch_size = 64  # size of the minibatch for stochastic gradient descent (or Adam)
valid_size = 0.2  # fraction of the training data to reserve for validation
num_epochs = 70  # number of epochs for training
num_classes = 57  # number of classes. Do not change this
dropout = 0.1 # for the original model
learning_rate = 0.001  # Learning rate for SGD (or Adam)
opt = 'adam'      # optimizer. 'sgd' or 'adam'
weight_decay = 0.001 # regularization. Increase this to combat overfitting


def main() -> None:
    """The entry point for this script."""
    
    # Get the information about how this script was distributed.
    # PyTorch doesn't automatically establish communication, so we have to do it ourselves.
    setup = SlurmSetup()
    print(f'Rank {setup.rank}: starting up.')
    setup.establish_communication()
    print(f'Rank {setup.rank}: communication is ready.')

    # All processes create a data loader from our custom quadratic dataset and a distributed sampler.
    # We'll force PyTorch to use a specific seed so all instances of the script generate the same data.
    torch.manual_seed(0)

    data_loaders = get_data_loaders(world_size=setup.world_size, rank=setup.rank, batch_size=batch_size)

    # model_transfer = get_model_transfer_learning("resnet18")
    models_list = ['resnet18', 'vgg16', 'mobilenet_v3_small', 'original']

    for model_name in models_list:
        if model_name == 'original':
            model = Model(num_classes, dropout)
        else:
            model = get_models(model_name)

        model = DDP(model).to('cuda')

        optimizer = get_optimizer(
            model,
            learning_rate=learning_rate,
            optimizer=opt,
            weight_decay=weight_decay,
        )
        loss = get_loss()

        # All processes create an instance of our model and wrap it in Distributed Data Parallel,
        # which will handle all the distributed communication for us.
        # We'll put it on the GPU with the same ID as this process's local rank (the rank on this node).
        if setup.is_main_process():
            print(f"=======================================STARTING OPTIMIZATION FOR {model_name} AT {datetime.utcnow()}=============================================")

        print(f'=================================RANK {setup.rank}: STARTED TRAINING AT {datetime.utcnow()}.=====================================================')
        optimize(
            data_loaders,
            model,
            optimizer,
            loss,
            n_epochs=num_epochs,
            save_path=f"checkpoints/model_transfer_{model_name}_{num_epochs}.pt",
            interactive_tracking=False,
            setup=setup
        )


        # Evaluate the model using a brand new dataset.
        # We'll only do this on the main node, that way we don't have to combine several loss outputs.
        if setup.is_main_process():
            test_loss = one_epoch_test(data_loaders['test'], model, loss)

            print(f'Rank {setup.rank}: The final loss is: {test_loss}.')

        print(f'===========================================RANK {setup.rank}: DONE TRAINING AT {datetime.utcnow()}.=====================================')
        if setup.is_main_process():
            print(f"=======================================TEST COMPLETED FOR {model_name} AT {datetime.utcnow()}=============================================")


if __name__ == '__main__':
    main()
