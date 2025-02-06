import os
import ogb
import dgl
import torch
from tqdm import tqdm
from ogb.nodeproppred import DglNodePropPredDataset
from dgl.dataloading import MultiLayerFullNeighborSampler, MultiLayerNeighborSampler
from torch.utils.data.dataloader import DataLoader
from models import SAGE
from dataset import NodeSet, NbrSampleCollater
import logging
import argparse
import time
from tensorboardX import SummaryWriter
from utils import *

from eva_iterator import EVAIterator
EVA_ITERATOR_SAMPLE_DURATION = 30
EVA_ITERATOR_TEST_MODE = False
EVA_ITERATOR_TEST_MODE_LOG_PERIOD = 60

class Trainer(object):
    def __init__(self, data, args, log_path):
        self.args = args
        self.log_path = log_path
        self.graph, self.labels, self.train_idx, self.valid_idx, self.test_idx = data

        self.fanouts_train = list(map(int, args.fanouts_train.split(',')))
        self.fanouts_valid = list(map(int, args.fanouts_valid.split(','))) if args.fanouts_valid is not None \
            else self.fanouts_train
        self.fanouts_test = list(map(int, args.fanouts_test.split(','))) if args.fanouts_test is not None \
            else self.fanouts_train
        assert len(self.fanouts_train) == len(self.fanouts_valid) == len(self.fanouts_test)

        # keep trying until the device is available
        self.device = torch.device('cuda')
        # self.device = torch.device('cpu')
        self.sage = SAGE(128, 1024, 172, len(self.fanouts_train), torch.nn.functional.leaky_relu, 0.1).to(
            self.device)

        self.batch_size = 1024
        self.num_epoch = args.num_epoch
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.global_iter = 0
        self.start_from_epoch = 0
        self.start_from_iter = 0 # not used
        self.cur_epoch = 0
        self.cur_iter = 0

        self.optimizer = torch.optim.Adam(self.sage.parameters(), lr=1e-3)
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=self.optimizer, factor=0.8,
                                                                      patience=1000, verbose=True)

        self.tb_writer = SummaryWriter(f'{log_path}/tensorboard')
        self.log_file = open(f'{log_path}/log.txt', mode='w', buffering=1)
        write_dict(args.__dict__, self.log_file)
        write_dict(self.sage.profile, self.log_file)
        self.num_parameters = count_parameters(self.sage)
        self.log_file.write(f'num_parameters={self.num_parameters}\n')
        self.valid_loss = []
        self.valid_accu = []

        # dataloader
        print('preparing dataloader', flush=True)
        self.train_collater = NbrSampleCollater(
            self.graph, MultiLayerNeighborSampler(fanouts=self.fanouts_train, replace=False))
        print('train_collater prepared', flush=True)
        self.train_node_set = NodeSet(self.train_idx.tolist(), self.labels[self.train_idx].tolist())
        print('train_node_set prepared', flush=True)
        self.train_node_loader = DataLoader(dataset=self.train_node_set, batch_size=self.batch_size,
                                            shuffle=True, num_workers=8, pin_memory=True,
                                            collate_fn=self.train_collater.collate, drop_last=False)
        self.train_node_loader = EVAIterator(
            self.train_node_loader,
            sample_duration=EVA_ITERATOR_SAMPLE_DURATION,
            test_mode=EVA_ITERATOR_TEST_MODE,
            test_mode_log_period=EVA_ITERATOR_TEST_MODE_LOG_PERIOD
        )
        print('train_node_loader prepared', flush=True)

        self.valid_collater = NbrSampleCollater(
            self.graph, MultiLayerNeighborSampler(fanouts=self.fanouts_valid, replace=False))
        self.valid_node_set = NodeSet(self.valid_idx.tolist(), self.labels[self.valid_idx].tolist())
        self.valid_node_loader = DataLoader(dataset=self.valid_node_set, batch_size=self.batch_size,
                                            shuffle=False, num_workers=0, pin_memory=True,
                                            collate_fn=self.valid_collater.collate, drop_last=False)
        self.test_collater = NbrSampleCollater(
            self.graph, MultiLayerNeighborSampler(fanouts=self.fanouts_test, replace=False))
        self.test_node_set = NodeSet(self.test_idx.tolist(), self.labels[self.test_idx].tolist())
        self.test_node_loader = DataLoader(dataset=self.test_node_set, batch_size=8,
                                           shuffle=False, num_workers=0, pin_memory=True,
                                           collate_fn=self.test_collater.collate, drop_last=False)

    def run(self):
        # load checkpoint
        if os.path.exists(f'latest.ckpt'):
            self.load_ckpt(f'latest.ckpt')
            self.log_file.write('load latest checkpoint\n')
            print('load latest checkpoint', flush=True)
        
        for epoch in range(self.num_epoch):
            if epoch < self.start_from_epoch:
                continue
            self.train(epoch, self.optimizer, self.lr_scheduler)
            # self.valid()
        # best_epoch = torch.topk(torch.tensor(self.valid_loss), k=1, largest=False).indices.view(-1).item()
        # self.sage.load_state_dict(torch.load(f'{self.log_path}/ckpt/epoch{best_epoch}.ckpt'))
        # self.log_file.write(f'inference with checkpoint of epoch {best_epoch}\n')
        # test_accu = self.test()
        # test_accu = 0
        self.tb_writer.close()
        self.log_file.close()
        # return self.valid_accu[best_epoch], test_accu
        return 0, 0

    def train(self, epoch, optimizer, lr_scheduler, iter_per_log = 20):
        self.sage.train()
        start_time = time.time()
        self.cur_epoch = epoch
        print(f"start_epoch: {epoch}", flush=True)
        # print(f"start_iter: {self.start_from_iter}", flush=True)
        times = []
        for n_iter, (blocks, labels) in enumerate(tqdm(self.train_node_loader, desc=f'train epoch {epoch}')):
            # if epoch == self.start_from_epoch and n_iter < self.start_from_iter:
            #     print(f"skip iter {n_iter}", flush=True)
            #     continue
            # if n_iter == 0:
            #     print(f"{epoch}th epoch: {time.time()}", flush=True)
            # blocks = torch.from_numpy(np.asarray(blocks))
            # print("blocks")
            # print(blocks)
            # print(len(blocks))
            # print(blocks)
            # print("labels")
            # print(labels)
            # og
            # blocks = blocks[-1]
            # blocks = [block.to(self.device) for block in blocks]
            blocks = [block.to(self.device,non_blocking=True) for block in blocks[-1]]
            labels = labels.to(self.device,non_blocking=True)
            batch_size = labels.shape[0]
            outputs = self.sage(blocks)
            pred = torch.topk(outputs, k=1).indices.view(-1)
            loss = self.loss_fn(outputs, labels)
            accu = self.count_corrects(pred, labels) / labels.shape[0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step(loss)
            self.global_iter += 1
            loss_item = loss.item() / batch_size
            self.log_file.write(
                f'epoch{epoch}\titer {n_iter}\t{n_iter * 100 / len(self.train_node_loader):.4f}%\tlr={lr_scheduler._last_lr}\tloss={loss_item:.8f}\taccu={accu:.6f}\n')
            self.tb_writer.add_scalar('train/lr', lr_scheduler._last_lr, self.global_iter)
            self.tb_writer.add_scalar('train/loss', loss_item, self.global_iter)
            self.tb_writer.add_scalar('train/accu', accu, self.global_iter)

            self.cur_iter = n_iter

            if self.global_iter % iter_per_log == 0:
                if self.global_iter != iter_per_log:
                    times.append(time.time() - start_time)
                start_time = time.time()
                if len(times) > 0:
                    print(f"avg time per {iter_per_log} iters = {sum(times)/len(times)}. Latest time = {times[-1]}", flush=True)

        # finish training, save checkpoint
        self.save_ckpt()

    def valid(self):
        self.sage.eval()
        correct_cnt = 0
        loss_item = 0
        with torch.no_grad():
            for blocks, labels in self.valid_node_loader:
                blocks = blocks[-1]
                blocks = [block.to(self.device) for block in blocks]
                labels = labels.to(self.device)
                batch_size = labels.shape[0]
                outputs = self.sage(blocks)
                loss = self.loss_fn(outputs, labels)
                loss_item += loss.item() / batch_size
                pred = torch.topk(outputs, k=1).indices.view(-1)
                correct_cnt += self.count_corrects(pred, labels)
        avg_loss_amount = loss_item / len(self.valid_node_loader)
        accu = correct_cnt / len(self.valid_node_set)
        self.log_file.write(f'valid avg_loss={avg_loss_amount:.8f} accu={accu:.6f}\n')
        self.tb_writer.add_scalar('valid/loss', avg_loss_amount, self.global_iter)
        self.tb_writer.add_scalar('valid/accu', accu, self.global_iter)
        self.valid_loss.append(avg_loss_amount)
        self.valid_accu.append(accu)

    def test(self):
        self.sage.eval()
        correct_cnt = 0
        loss_amount = 0
        with torch.no_grad():
            for blocks, labels in tqdm(self.test_node_loader, desc='test'):
                blocks = blocks[-1]
                blocks = [block.to(self.device) for block in blocks]
                labels = labels.to(self.device)
                batch_size = labels.shape[0]
                outputs = self.sage(blocks)
                loss = self.loss_fn(outputs, labels)
                loss_amount += loss.item() / batch_size
                pred = torch.topk(outputs, k=1).indices.view(-1)
                correct_cnt += self.count_corrects(pred, labels)
        avg_loss_amount = loss_amount / len(self.test_node_loader)
        accu = correct_cnt / len(self.test_node_set)
        self.log_file.write(
            f'test avg_loss={avg_loss_amount:.8f} accu={accu:.6f}\n')
        self.tb_writer.add_scalar('test/loss', avg_loss_amount, self.global_iter)
        self.tb_writer.add_scalar('test/accu', accu, self.global_iter)
        return accu

    def save_ckpt(self):
        print(f'save checkpoint at epoch {self.cur_epoch} iter {self.cur_iter}', flush=True)
        state = {
            'model': self.sage.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict(),
            'epoch': self.cur_epoch,
            'iter': self.cur_iter,
        }

        torch.save(state, f'latest.ckpt')

    def load_ckpt(self, path):
        ckpt = torch.load(path)
        self.sage.load_state_dict(ckpt['model'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.lr_scheduler.load_state_dict(ckpt['lr_scheduler'])
        self.start_from_epoch = ckpt['epoch'] + 1
        self.start_from_iter = ckpt['iter'] # not used
        # self.sage.load_state_dict(torch.load(path))

    @staticmethod
    def count_corrects(pred: torch.Tensor, label: torch.Tensor) -> int:
        assert pred.dim() == 1 and label.dim() == 1 and pred.shape == label.shape
        return ((pred == label) + 0.0).sum().item()


def graphdata_preprocess(dgldataset: ogb.nodeproppred.DglNodePropPredDataset):
    graph, labels = dgldataset[0]
    srcs, dsts = graph.all_edges()
    graph.add_edges(dsts, srcs)
    labels = labels.view(-1).type(torch.int)
    splitted_idx = dgldataset.get_idx_split()
    train_idx, val_idx, test_idx = splitted_idx["train"], splitted_idx["valid"], splitted_idx["test"]
    return [graph, labels, train_idx, val_idx, test_idx]


if __name__ == '__main__':
    # see if gpu is available
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    assert torch.cuda.is_available()
    print(f"start main at time {time.time()}")
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', type=str, default='sage')
    parser.add_argument('--fanouts-train', type=str, default='12,12,12')
    parser.add_argument('--fanouts-valid', type=str, default=None)
    parser.add_argument('--fanouts-test', type=str, default='100,100,100')
    parser.add_argument('--dataset-root', type=str, default='/datasets')
    parser.add_argument('--num_epoch', type=int, default=1000)
    args = parser.parse_args()

    time_stamp = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
    valid_accus = []
    test_accus = []

    print(f'loading data from {args.dataset_root}', flush=True)
    start_time = time.time()
    data = graphdata_preprocess(
        #DglNodePropPredDataset('ogbn-papers100M', root=args.dataset_root)
        # DglNodePropPredDataset('ogbn-products', root=args.dataset_root)
        DglNodePropPredDataset('ogbn-arxiv', root=args.dataset_root)
        #DglNodePropPredDataset('ogbn-arxiv', root=os.path.join(os.environ['HOME'], 'data', 'OGB'))
    )
    print('data loaded', flush=True)
    print(f'elapsed time: {time.time() - start_time:.4f}s', flush=True)
    for seed in range(1):

        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        dgl.seed(seed)
        dgl.random.seed(seed)

        log_path = f'log/{args.version}-{time_stamp}-seed{seed}'

        if not os.path.exists('log'):
            os.mkdir('log')
        if not os.path.exists(log_path):
            os.mkdir(log_path)
            os.mkdir(os.path.join(log_path, 'ckpt'))
            backup_code(log_path)
        else:
            raise ValueError(f'log path {log_path} exists')

        trainer = Trainer(data, args, log_path)
        valid_accu, test_accu = trainer.run()
        valid_accus.append(valid_accu)
        test_accus.append(test_accu)
        print(f"runned 10 times")
        print(f"valid accus: {valid_accus}")
        print(f"test  accus: {test_accus}")
        print(f"average valid accu: {np.mean(valid_accus):.4f} ± {np.std(valid_accus):.4f}")
        print(f"average test  accu: {np.mean(test_accus):.4f} ± {np.std(test_accus):.4f}")
        print(f'numbers of params: {trainer.num_parameters}')
