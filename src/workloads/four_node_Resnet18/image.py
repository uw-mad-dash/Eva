import argparse
import copy
import os
import random
import shutil
import time
import warnings
import socket
from enum import Enum
import datetime

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Subset
from eva_iterator import EVAIterator
EVA_SAMPLE_DURATION = 300
EVA_ITERATOR_TEST_MODE = False 
EVA_ITERATOR_TEST_MODE_LOG_PERIOD = 60


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', metavar='DIR', nargs='?', default='imagenet',
                    help='path to dataset (default: imagenet)')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--start-batch', default=0, type=int, metavar='N',
                    help='manual batch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--dummy', action='store_true', help="use fake data to benchmark")
parser.add_argument('--checkpoint-dir', default='./', type=str,
                    help='checkpoint directory')
parser.add_argument('--checkpoint-freq', default=100, type=int,
                    help='save checkpoint every N batches')

best_acc1 = 0

def is_master_up(master_address, master_port, timeout=5):
    """Check if the master node is up by trying to connect to it."""
    # check if i'm the master
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        try:
            sock.connect((master_address, master_port))
            return True
        except socket.error:
            return False
        


class CustomImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None, target_transform=None):
        super(CustomImageFolder, self).__init__(root, transform, target_transform)
        self.original_samples = copy.deepcopy(self.samples)
        self.original_targets = copy.deepcopy(self.targets)
    
    def make_dataset(self, dir, class_to_idx, extensions=None, is_valid_file=None):
        # check if there is a cache file that we can just load
        cache_file = os.path.join(dir, "samples.txt")
        if os.path.exists(cache_file):
            print("loading samples from cache", flush=True)
            with open(cache_file, "r") as f:
                samples = f.read().splitlines()
            samples = [tuple(sample.split(" ")) for sample in samples]
            # class index is a string, so we need to convert it to int
            samples = [(sample[0], int(sample[1])) for sample in samples]
        else:
            # create the cache file
            print("creating cache file", flush=True)
            samples = super(CustomImageFolder, self).make_dataset(dir, class_to_idx, extensions, is_valid_file)
            with open(cache_file, "w") as f:
                f.write("\n".join([" ".join(map(str, sample)) for sample in samples]))
            
        return samples
    
    def find_classes(self, dir):
        # check if there is a cache file that we can just load
        cache_file = os.path.join(dir, "classes.txt")
        if os.path.exists(cache_file):
            print("loading classes from cache", flush=True)
            with open(cache_file, "r") as f:
                classes = f.read().splitlines()
            class_to_idx = {classes[i]: i for i in range(len(classes))}
        else:
            print("creating cache file", flush=True)
            # create the cache file
            classes, class_to_idx = super(CustomImageFolder, self).find_classes(dir)
            with open(cache_file, "w") as f:
                f.write("\n".join(classes))

        return classes, class_to_idx
    
    def __getitem__(self, index):
        try:
            return super(CustomImageFolder, self).__getitem__(index)
        except Exception as e:
            # return some random data
            print(f"Error: {e}. Just giving some random data.", flush=True)
            return torch.rand(3, 224, 224), 0

    def set_start_index(self, start_index):
        if start_index >= len(self.original_samples):
            print(f"start_index {start_index} is greater than the number of samples {len(self.original_samples)}. Setting start_index to last index.", flush=True)
            start_index = len(self.original_samples) - 1
        self.samples = self.original_samples[start_index:]
        self.targets = self.original_targets[start_index:]

# def get_custom_loader(dataset, batch_size, workers, start_batch=0):

#     if distributed:
#         sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False, drop_last=True)
#     else:
#         sampler = None

#     loader = torch.utils.data.DataLoader(
#         dataset, batch_size=batch_size, shuffle=False,
#         num_workers=workers, pin_memory=True, sampler=sampler)

#     train_loader = torch.utils.data.DataLoader(
#         train_dataset, batch_size=batch_size, shuffle=False,
#         num_workers=workers, pin_memory=True, sampler=train_sampler)
#     val_loader = torch.utils.data.DataLoader(
#         val_dataset, batch_size=batch_size, shuffle=False,
#         num_workers=workers, pin_memory=True, sampler=val_sampler)

#     # Calculate the start index based on the start epoch and batch
#     start_index = start_batch * batch_size
#     train_dataset.set_start_index(start_index)

#     return train_loader, val_loader

def main():
    print(f"Hi! You've entered main", flush=True)
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    print(f"args: {args}", flush=True)
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    if torch.cuda.is_available():
        ngpus_per_node = torch.cuda.device_count()
    else:
        ngpus_per_node = 1
    print(f"ngpus_per_node: {ngpus_per_node}", flush=True)
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu
    start_time = time.time()

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu), flush=True)

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        print(f"args.rank: {args.rank}", flush=True)
        dist_url = args.dist_url.replace("tcp://", "")
        master_address, master_port = dist_url.split(":")
        master_port = int(master_port)

        print(f"args.rank: {args.rank}", flush=True)
        print(f"Waiting for the master node at {master_address}:{master_port}", flush=True)

        os.environ['NCCL_SOCKET_IFNAME'] = "eth1"

        # Wait until the master node is up
        while args.rank != 0 and not is_master_up(master_address, master_port):
            print("Master not up yet, retrying...", flush=True)
            time.sleep(1)  # wait before checking again

        print("Master node is up, initializing process group", flush=True)
        timeout = datetime.timedelta(hours=3)
        dist.init_process_group(backend="nccl", init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank,
                                timeout=timeout)
        print("Process group initialized", flush=True)
    
    # find the network adapter that has the master node's IP address


    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch), flush=True)
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch), flush=True)
        model = models.__dict__[args.arch]()

    if not torch.cuda.is_available() and not torch.backends.mps.is_available():
        print('using CPU, this will be slow', flush=True)
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if torch.cuda.is_available():
            if args.gpu is not None:
                torch.cuda.set_device(args.gpu)
                model.cuda(args.gpu)
                # When using a single GPU per process and per
                # DistributedDataParallel, we need to divide the batch size
                # ourselves based on the total number of GPUs of the current node.
                args.batch_size = int(args.batch_size / ngpus_per_node)
                args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            else:
                model.cuda()
                # DistributedDataParallel will divide and allocate batch_size to all
                # available GPUs if device_ids are not set
                model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        model = model.to(device)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    if torch.cuda.is_available():
        if args.gpu:
            device = torch.device('cuda:{}'.format(args.gpu))
        else:
            device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    # define loss function (criterion), optimizer, and learning rate scheduler
    criterion = nn.CrossEntropyLoss().to(device)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume), flush=True)
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            elif torch.cuda.is_available():
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            args.start_batch = checkpoint.get('batch', 0)  # Default to 0 if 'batch' is not in checkpoint
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = torch.tensor(best_acc1).to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            print(f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']}, batch {args.start_batch})")
        else:
            print("=> no checkpoint found at '{}'".format(args.resume), flush=True)


    # Data loading code
    if args.dummy:
        print("=> Dummy data is used!", flush=True)
        train_dataset = datasets.FakeData(1281167, (3, 224, 224), 1000, transforms.ToTensor())
        print("=> train dataset done", flush=True)
        val_dataset = datasets.FakeData(50000, (3, 224, 224), 1000, transforms.ToTensor())
        print("=> val dataset done", flush=True)
    else:
        traindir = os.path.join(args.data, 'train')
        valdir = os.path.join(args.data, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

        print("=> using real data", flush=True)
        train_dataset = CustomImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))
        print("=> train dataset done", flush=True)

        val_dataset = CustomImageFolder(
            valdir,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]))

    # if args.distributed:
    #     train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=False, drop_last=False)
    #     val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, drop_last=False)
    # else:
    #     train_sampler = None
    #     val_sampler = None

    print("=> train loader", flush=True)
    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset, batch_size=args.batch_size, shuffle=False,
    #     num_workers=args.workers, pin_memory=True, sampler=train_sampler)
    train_loader = EVAIterator(
        None,
        sample_duration=EVA_SAMPLE_DURATION,
        test_mode=EVA_ITERATOR_TEST_MODE,
        test_mode_log_period=EVA_ITERATOR_TEST_MODE_LOG_PERIOD
    )
    
    print("=> val loader", flush=True)

    # val_loader = torch.utils.data.DataLoader(
    #     val_dataset, batch_size=args.batch_size, shuffle=False,
    #     num_workers=args.workers, pin_memory=True, sampler=val_sampler)

    # if args.evaluate:
    #     validate(val_loader, model, criterion, device, args)
    #     return

    print("start training at time {}".format(time.time() - start_time), flush=True)
    for epoch in range(args.start_epoch, args.epochs):
        if epoch == args.start_epoch:
            print(f"skipping to batch {args.start_batch}", flush=True)
            start_index = args.start_batch * args.batch_size * args.world_size
            train_dataset.set_start_index(start_index)
            start_batch = args.start_batch
        else:
            train_dataset.set_start_index(0)
            start_batch = 0

        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=False, drop_last=False)
            train_sampler.set_epoch(epoch)
        else:
            train_sampler = None
        train_loader._data_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True, sampler=train_sampler)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, scheduler, epoch, start_batch, device, args)

        # evaluate on validation set
        #acc1 = validate(val_loader, model, criterion, device, args)
        acc1=best_acc1

        scheduler.step()

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not args.distributed or args.rank % args.world_size == 0:
            save_checkpoint({
                'epoch': epoch+1,
                'batch': 0,  
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, False, args.checkpoint_dir)


def train(train_loader, model, criterion, optimizer, scheduler, epoch, start_batch, device, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    before_loop_start_time = time.time()
    for i, (images, target) in enumerate(train_loader):
        print(f"epoch {epoch} batch {start_batch + i} at time {time.time() - before_loop_start_time}", flush=True)
        # measure data loading time
        data_time.update(time.time() - end)

        # move data to the same device as model
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i+1) % args.print_freq == 0:
            progress.display(i + 1)

        if (i+1) % args.checkpoint_freq == 0 and (not args.distributed or args.rank % args.world_size == 0):
            save_checkpoint({
                'epoch': epoch,
                'batch': start_batch + i + 1,  # Save the next batch to start from
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, False, args.checkpoint_dir)
        
def validate(val_loader, model, criterion, device, args):

    def run_validate(loader, base_progress=0):
        with torch.no_grad():
            end = time.time()
            for i, (images, target) in enumerate(loader):
                i = base_progress + i
                images = images.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                # if args.gpu is not None and torch.cuda.is_available():
                #     images = images.cuda(args.gpu, non_blocking=True)
                # if torch.backends.mps.is_available():
                #     images = images.to('mps')
                #     target = target.to('mps')
                # if torch.cuda.is_available():
                #     target = target.cuda(args.gpu, non_blocking=True)

                # compute output
                output = model(images)
                loss = criterion(output, target)

                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % args.print_freq == 0:
                    progress.display(i + 1)

    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        len(val_loader) + (args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset))),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    run_validate(val_loader)
    if args.distributed:
        top1.all_reduce()
        top5.all_reduce()

    if args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset)):
        aux_val_dataset = Subset(val_loader.dataset,
                                 range(len(val_loader.sampler) * args.world_size, len(val_loader.dataset)))
        aux_val_loader = torch.utils.data.DataLoader(
            aux_val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
        run_validate(aux_val_loader, len(val_loader))

    progress.display_summary()

    return top1.avg


def save_checkpoint(state, is_best, checkpoint_dir, filename='checkpoint.pth.tar'):
    checkpoint_path = os.path.join(checkpoint_dir, filename)
    print(f"saving checkpoint at {checkpoint_path}...", flush=True)
    torch.save(state, f"{checkpoint_path}.tmp")
    os.rename(f"{checkpoint_path}.tmp", checkpoint_path)
    if is_best:
        best_model_path = os.path.join(checkpoint_dir, "model_best.pth.tar")
        shutil.copyfile(checkpoint_path, best_model_path)

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)

        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries), flush=True)

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries), flush=True)

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
