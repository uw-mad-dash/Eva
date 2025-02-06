import argparse
import os
import socket

import time
import dgl
import dgl.nn as dglnn
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from argo import ARGO
from dgl.data import (
    AsNodePredDataset,
    FlickrDataset,
    RedditDataset,
    YelpDataset,
)
from dgl.dataloading import DataLoader, NeighborSampler, ShaDowKHopSampler
from ogb.nodeproppred import DglNodePropPredDataset
from torch.nn.parallel import DistributedDataParallel

from eva_iterator import EVAIterator
EVA_SAMPLE_DURATION = 300
EVA_ITERATOR_TEST_MODE = False
EVA_ITERATOR_TEST_MODE_LOG_PERIOD = 60


def find_free_port():
    """Find a free port."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('', 0))
    port = s.getsockname()[1]
    s.close()
    return port


class GNN(nn.Module):
    def __init__(
        self, in_size, hid_size, out_size, num_layers=3, model_name="sage"
    ):
        super().__init__()
        self.layers = nn.ModuleList()

        # GraphSAGE-mean
        if model_name.lower() == "sage":
            self.layers.append(dglnn.SAGEConv(in_size, hid_size, "mean"))
            for i in range(num_layers - 2):
                self.layers.append(dglnn.SAGEConv(hid_size, hid_size, "mean"))
            self.layers.append(dglnn.SAGEConv(hid_size, out_size, "mean"))
        # GCN
        elif model_name.lower() == "gcn":
            kwargs = {
                "norm": "both",
                "weight": True,
                "bias": True,
                "allow_zero_in_degree": True,
            }
            self.layers.append(dglnn.GraphConv(in_size, hid_size, **kwargs))
            for i in range(num_layers - 2):
                self.layers.append(
                    dglnn.GraphConv(hid_size, hid_size, **kwargs)
                )
            self.layers.append(dglnn.GraphConv(hid_size, out_size, **kwargs))
        else:
            raise NotImplementedError

        self.dropout = nn.Dropout(0.5)
        self.hid_size = hid_size
        self.out_size = out_size

    def forward(self, blocks, x):
        h = x
        if hasattr(blocks, "__len__"):
            for l, (layer, block) in enumerate(zip(self.layers, blocks)):
                h = layer(block, h)
                if l != len(self.layers) - 1:
                    h = F.relu(h)
                    h = self.dropout(h)
        else:
            for l, layer in enumerate(self.layers):
                h = layer(blocks, h)
                if l != len(self.layers) - 1:
                    h = F.relu(h)
                    h = self.dropout(h)
        return h


def _train(**kwargs):
    # print(f"rank {kwargs['rank']} in _train", flush=True)
    total_loss = 0
    loader = kwargs["loader"]
    model = kwargs["model"]
    opt = kwargs["opt"]
    load_core = kwargs["load_core"]
    comp_core = kwargs["comp_core"]

    device = torch.device("cpu")
    # print(f"rank {kwargs['rank']} enabling cpu affinity", flush=True)
    with loader.enable_cpu_affinity(
        loader_cores=load_core, compute_cores=comp_core
    ):
        # print(f"rank {kwargs['rank']} cpu affinity enabled", flush=True)
        for it, (input_nodes, output_nodes, blocks) in enumerate(loader):
            if hasattr(blocks, "__len__"):
                x = blocks[0].srcdata["feat"].to(torch.float32)
                y = blocks[-1].dstdata["label"]
            else:
                x = blocks.srcdata["feat"].to(torch.float32)
                y = blocks.dstdata["label"]
            if kwargs["device"] == "cpu":  # for papers100M
                y = y.type(torch.LongTensor)
                y_hat = model(blocks, x)
            else:
                y = y.type(torch.LongTensor).to(device)
                y_hat = model(blocks, x).to(device)
            try:
                loss = F.cross_entropy(
                    y_hat[: output_nodes.shape[0]], y[: output_nodes.shape[0]]
                )
            except:
                loss = F.binary_cross_entropy_with_logits(
                    y_hat[: output_nodes.shape[0]].float(),
                    y[: output_nodes.shape[0]].float(),
                    reduction="sum",
                )
            opt.zero_grad()
            loss.backward()
            opt.step()
            del input_nodes, output_nodes, blocks
            total_loss += loss.item()
    return total_loss


def train(
    args, g, data, rank, world_size, comp_core, load_core, counter, b_size, ep
):
    print("Training on rank", rank, flush=True)
    start_time = time.time()
    num_classes, train_idx = data
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    device = torch.device("cpu")
    hidden = args.hidden
    # create GraphSAGE model
    in_size = g.ndata["feat"].shape[1]
    model = GNN(
        in_size,
        hidden,
        num_classes,
        num_layers=args.layer,
        model_name=args.model,
    ).to(device)
    model = DistributedDataParallel(model)
    num_of_samplers = len(load_core)
    # create loader
    drop_last, shuffle = True, True
    if args.sampler.lower() == "neighbor":
        sampler = NeighborSampler(
            [int(fanout) for fanout in args.fan_out.split(",")],
            prefetch_node_feats=["feat"],
            prefetch_labels=["label"],
        )
        assert len(sampler.fanouts) == args.layer
    elif args.sampler.lower() == "shadow":
        sampler = ShaDowKHopSampler(
            [10, 5],
            output_device=device,
            prefetch_node_feats=["feat"],
        )
    else:
        raise NotImplementedError

    train_dataloader = DataLoader(
        g,
        train_idx.to(device),
        sampler,
        device=device,
        batch_size=b_size,
        drop_last=drop_last,
        shuffle=shuffle,
        num_workers=num_of_samplers,
        use_ddp=True,
    )
    if rank == 0:
        print(f"creating eva iterator", flush=True)
        train_dataloader = EVAIterator(
            train_dataloader,
            sample_duration=EVA_SAMPLE_DURATION,
            test_mode=EVA_ITERATOR_TEST_MODE,
            test_mode_log_period=EVA_ITERATOR_TEST_MODE_LOG_PERIOD
        )
        print(f"eva iterator created", flush=True)

    # training loop
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    params = {
        # training
        "loader": train_dataloader,
        "model": model,
        "opt": opt,
        # logging
        "rank": rank,
        "train_size": len(train_idx),
        "batch_size": b_size,
        "device": device,
        "process": world_size,
    }

    PATH = "model.pt"
    if counter[0] != 0:
        print("loading model", flush=True)
        checkpoint = torch.load(PATH)
        model.load_state_dict(checkpoint["model_state_dict"])
        opt.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch = checkpoint["epoch"]
        loss = checkpoint["loss"]
    
    actual_epoch = counter[0]

    for epoch in range(actual_epoch, actual_epoch + ep):
        # if rank == 0:
        #     print("Epoch:", epoch, flush=True)
        start_time = time.time()
        params["epoch"] = epoch
        model.train()
        params["load_core"] = load_core
        params["comp_core"] = comp_core
        # print(f"rank {rank} about to _train", flush=True)
        loss = _train(**params)
        if rank == 0:
            print(f"Finished epoch {epoch} out of {actual_epoch + ep - 1} in {time.time() - start_time} seconds", flush=True)
            print("loss:", loss)
            torch.save( 
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                    "loss": loss,
                },
                "model.tmp"
            )
            os.rename("model.tmp", PATH)
            

    dist.barrier()
    # EPOCH = counter[0]
    # LOSS = loss
    # if rank == 0:
    #     print(f"Took {time.time() - start_time} seconds", flush=True)
    #     torch.save(
    #         {
    #             "epoch": EPOCH,
    #             "model_state_dict": model.state_dict(),
    #             "optimizer_state_dict": opt.state_dict(),
    #             "loss": LOSS,
    #         },
    #         PATH,
    #     )
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="ogbn-products",
        choices=[
            "ogbn-papers100M",
            "ogbn-products",
            "reddit",
            "yelp",
            "flickr",
        ],
    )
    parser.add_argument("--n_search", type=int, default=1)
    parser.add_argument("--cpu_count", type=int, default=8)
    parser.add_argument("--epoch", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=1024 * 16)
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--fan_out", type=str, default="15,10,5")
    parser.add_argument(
        "--sampler",
        type=str,
        default="neighbor",
        choices=["neighbor", "shadow"],
    )
    parser.add_argument(
        "--model", type=str, default="sage", choices=["sage", "gcn"]
    )
    parser.add_argument("--hidden", type=int, default=128)
    arguments = parser.parse_args()

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

    if arguments.dataset in ["reddit", "flickr", "yelp"]:
        if arguments.dataset == "reddit":
            dataset = RedditDataset()
        elif arguments.dataset == "flickr":
            dataset = FlickrDataset()
        else:
            dataset = YelpDataset()
        g = dataset[0]
        train_mask = g.ndata["train_mask"]
        idx = []
        for i in range(len(train_mask)):
            if train_mask[i]:
                idx.append(i)
        dataset.train_idx = torch.tensor(idx)
    else:
        dataset = AsNodePredDataset(DglNodePropPredDataset(arguments.dataset, root="/datasets"))
        g = dataset[0]

    data = (dataset.num_classes, dataset.train_idx)

    in_size = g.ndata["feat"].shape[1]
    out_size = dataset.num_classes
    hidden_size = int(arguments.hidden)

    # see if there is checkpoint
    if os.path.exists("model.pt"):
        checkpoint = torch.load("model.pt")
        # get the epoch
        prev_epoch = checkpoint["epoch"]
    else:
        prev_epoch = -1
    print("starting from epoch", prev_epoch+1)

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(find_free_port())
    print("using port", os.environ["MASTER_PORT"])
    mp.set_start_method("fork", force=True)
    n_search = arguments.n_search
    total_epoch = arguments.epoch
    
    if prev_epoch + 1 >= total_epoch:
        print("Training is already done")
        exit()
    runtime = ARGO(n_search=n_search, cpu_count=arguments.cpu_count, epoch=total_epoch, batch_size=arguments.batch_size, counter=prev_epoch+1)
    runtime.run(train, args=(arguments, g, data))
