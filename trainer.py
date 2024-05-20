import torch
import torchvision
import torchvision.transforms as transforms
import time
import numpy as np
from torch import nn
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist
import os
from torch.distributed.optim import ZeroRedundancyOptimizer
from CustomDataset import CustomDataset
from mixup import Mixup
from auto_augment import RandAugment,rand_augment_ops
from random_erasing import RandomErasing
from torch_optimizer import Lamb

start_time = time.time()

class CustomCrossEntropy(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1, weight = None):
        """if smoothing == 0, it's one-hot method
           if 0 < smoothing < 1, it's smooth method
        """
        super(CustomCrossEntropy, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.weight = weight
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        assert 0 <= self.smoothing < 1
        pred = pred.log_softmax(dim=self.dim)

        if self.weight is not None:
            pred = pred * self.weight.unsqueeze(0)

        return torch.mean(torch.sum(-target * pred, dim=self.dim))

# Initialize transformations for data augmentation
transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    RandAugment(ops=rand_augment_ops(), num_layers=2),
    transforms.ToTensor(),
    # RandomErasing(probability=0.25,device=torch.device('cpu')),
    transforms.Normalize([0, 0, 0], [1, 1, 1])
])
Othertransform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0, 0, 0], [1, 1, 1])
])
# Load the ImageNet Object Localization Challenge dataset
train_dataset = torchvision.datasets.ImageFolder(
    root='/s/b/proj/gol_bitvec/imagenet',
    transform=transform
)
# Setting shuffle=False is intentional because trainning on 16 gpus at once needs to make sure the random samples dont over lap
# batcg_size//16 because the model will run 16 examples at once then aggregate gradients so you need to account for that

def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12345"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)

# Load the validation dataset
val_dataset = CustomDataset(csv_file= 'images.csv', root_dir = '/s/b/proj/gol_bitvec/val',transform = Othertransform)
# Train the model...
train_loss = []
test_accuracy = []
def Trainner(rank, world_size,num_epochs,batch_size,learning_rate):
    ddp_setup(rank, world_size)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size // world_size, shuffle=False,pin_memory=True,
                                               sampler=torch.utils.data.distributed.DistributedSampler(train_dataset), num_workers=4, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size//world_size, shuffle=False,pin_memory=True,
                                               sampler=torch.utils.data.distributed.DistributedSampler(val_dataset), num_workers=4)
    mixup = Mixup(mixup_alpha=.1, cutmix_alpha=1.0)
    model = torchvision.models.resnet50(pretrained=False)
    model = model.to(rank)
    model = DDP(model, device_ids=[rank], find_unused_parameters=False)

    model = torch.compile(model)
    
    # criterion = CustomCrossEntropy(classes=1_000,dim=1)
    criterion = nn.CrossEntropyLoss()
    record = tries = TotalTime = 0
    scaler = torch.cuda.amp.GradScaler()
    optimizer = ZeroRedundancyOptimizer(model.parameters(), optimizer_class=Lamb,lr=learning_rate,weight_decay=0.02)
    for epoch in range(num_epochs):
        # how the model knows to draw more data for an epoch that doesn't overlap
        train_loader.sampler.set_epoch(epoch)
        model.train()
        Epoch = epoch + 1
        for param_group in optimizer.param_groups:
            if Epoch <= 5:
                param_group['lr'] = Epoch*(learning_rate/5)
            else:
                param_group['lr'] = .5*learning_rate+.5*learning_rate*np.cos((Epoch-5)*np.pi/295)
        for inputs, labels in train_loader:

            inputs,labels = mixup(inputs,labels)
            optimizer.zero_grad()

            # Forward pass
            inputs=inputs.to(rank)
            labels =labels.to(rank)
            with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16) as autocast, torch.backends.cuda.sdp_kernel(
                    enable_flash=False) as disable:
                if rank == 2:
                    TemperFi = time.time()
                outputs = model(inputs)
                if rank == 2:
                    TotalTime += time.time() - TemperFi
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        # Print the loss for every epoch
        #     if rank == 1:
        if rank ==1:
            print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}',flush=True)
        model.eval()
        train_loss.append(loss)
        # Set the model to evaluation mode
        # Keep track of the total number of correct predictions and total number of samples
        total_correct = 0
        total_samples = 0

        # Iterate through the validation dataset
        with torch.no_grad():
            for _,inputs,labels in val_loader:
                with torch.cuda.amp.autocast(enabled=True,
                                             dtype=torch.float16) as autocast, torch.backends.cuda.sdp_kernel(
                        enable_flash=False) as disable:
                    outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)

                # Update the total number of correct predictions and total number of samples
                total_correct += (predicted == labels.to(rank)).sum()
                total_samples += labels.size(0)

        # Compute the accuracy
        accuracy = total_correct / total_samples
        dist.all_reduce(accuracy, op=dist.ReduceOp.SUM)
        accuracy = accuracy/16
        test_accuracy.append(accuracy)
        if rank ==1:
            print(f'Validation accuracy: {accuracy:.4f}')
    if rank == 1:
        print(f'Finished Training, Loss: {loss.item():.4f}')
        print(TotalTime,'This is very important')
    destroy_process_group()

if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    # world_size = 2
    batch_size = 2048
    learning_rate = 0.008
    epochs = 100
    mp.spawn(Trainner, args=(world_size, epochs,batch_size,learning_rate), nprocs=world_size, join=True)

    end_time = time.time()
    total_time = end_time - start_time
    print("Total running time:", total_time, "seconds")

    torch.save(train_loss, 'train_loss')
    torch.save(test_accuracy, 'test_accuracy')
