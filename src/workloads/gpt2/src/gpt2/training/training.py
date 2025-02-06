import tqdm
import os
import sys
import socket
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from gpt2.data import Dataset
from gpt2.training import TrainingSpec, TrainConfig, Recorder
from typing import Dict, Optional
import time

try:
    from apex import amp
except ModuleNotFoundError:
    pass

import warnings
warnings.filterwarnings(action='ignore')

from eva_iterator import EVAIterator
EVA_ITERATOR_SAMPLE_DURATION = 300
EVA_ITERATOR_TEST_MODE = False
EVA_ITERATOR_TEST_MODE_LOG_PERIOD = 60

def find_free_port():
    """Find a free port."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('', 0))
    port = s.getsockname()[1]
    s.close()
    return port

class Trainer(object):
    def __init__(self, spec: TrainingSpec, config: TrainConfig):
        self.spec = spec
        self.config = config
        self.port = find_free_port()

    def train(self,
              from_checkpoint: Optional[str] = None,
              from_pretrained: Optional[str] = None):
        if self.config.distributed:
            mp.spawn(self._train, args=(from_checkpoint, from_pretrained),
                     nprocs=self.config.gpus)
        else:
            self._train(0, from_checkpoint, from_pretrained)

    def _train(self,
               rank: int,
               from_checkpoint: Optional[str] = None,
               from_pretrained: Optional[str] = None):
        start_time = time.time()
        if self.config.distributed:
            print(f"rank {rank} initializing process group at port {self.port}", flush=True)
            torch.cuda.set_device(rank)
            dist.init_process_group(backend='nccl',
                                    init_method=f'tcp://127.0.0.1:{self.port}',
                                    world_size=self.config.gpus,
                                    rank=rank)

        # Initialize training environment and prepare datasets.
        self.spec.initialize()
        train_dataset, eval_dataset = self.spec.prepare_datasets()

        # Construct a model and load its pretrained weights.
        model = self.spec.construct_model().cuda()
        if from_pretrained:
            ckpt = torch.load(from_pretrained, map_location='cuda')
            model.load_state_dict(ckpt['model'])

            # Because the weights data allocates quite a lot of GPU memories,
            # we need to free the memories explicitly.
            del ckpt
            torch.cuda.empty_cache()

        # Create an optimizer and learning rate scheduler.
        optimizer, scheduler = self.spec.create_optimizer(model.parameters())
        recorder = Recorder()

        if self.config.use_amp:
            model, optimizer = amp.initialize(
                model, optimizer, opt_level='O2', verbosity=0)

        if self.config.distributed:
            model = nn.parallel.DistributedDataParallel(
                model, device_ids=[rank])

        start_step = 0
        # Restore last training states from checkpoint.
        if from_checkpoint:
            # check if the checkpoint exists
            if not os.path.exists(from_checkpoint):
                print(f"Checkpoint file not found: {from_checkpoint}, will start training from scratch.", flush=True)
            else:
                print(f"Loading checkpoint from {from_checkpoint}", flush=True)
                ckpt = torch.load(from_checkpoint, map_location='cuda')

                start_step = ckpt['step']
                recorder = ckpt['recorder']

                model.load_state_dict(ckpt['model'])
                optimizer.load_state_dict(ckpt['optimizer'])
                scheduler.load_state_dict(ckpt['scheduler'])

                train_dataset.assign(ckpt['train_dataset'])
                eval_dataset.assign(ckpt['eval_dataset'])

                if self.config.use_amp:
                    amp.load_state_dict(ckpt['amp'])

                # Because the checkpoint data allocates quite a lot of GPU
                # memories, we need to free the memories explicitly.
                del ckpt
                torch.cuda.empty_cache()

        if rank == 0:
            # Create tqdm iterator in master process to show the progress of
            # training.
            eva_iterator = EVAIterator(
                range(start_step + 1, self.config.total_steps),
                sample_duration=EVA_ITERATOR_SAMPLE_DURATION,
                wait_for_save_checkpoint=True,
                test_mode=EVA_ITERATOR_TEST_MODE,
                test_mode_log_period=EVA_ITERATOR_TEST_MODE_LOG_PERIOD
            )
            training_iters = tqdm.tqdm(
                eva_iterator,
                # range(start_step + 1, self.config.total_steps),
                total=self.config.total_steps,
                desc=self.config.description,
                dynamic_ncols=True, file=sys.stdout)
            training_iters.update(start_step + 1)
            start_time = time.time()
        else:
            # In other processes, use simple iterator rather than tqdm one.
            training_iters = range(start_step + 1, self.config.total_steps)

        start_time = time.time()
        for step in training_iters:
            # Clear CUDA cache which is used for training.
            torch.cuda.empty_cache()
            # if step == 1:
            #     print(f"Rank {rank} starting training at time {time.time()-start_time}", flush=True)

            recorder.record(
                self._train_step(rank, train_dataset, model, optimizer,
                                 scheduler),
                scope='train')

            # Clear CUDA cache which is used for evaluation.
            torch.cuda.empty_cache()

            if (step + 1) % self.config.eval_steps == 0:
                print(f"Rank {rank} evaluating at step {step}")
                recorder.record(
                    self._eval_step(rank, eval_dataset, model), scope='eval')
                recorder.stamp(step)

                if rank == 0:
                    training_iters.set_postfix_str(
                        recorder.format(self.config.log_format))

            # Save training states to checkpoint file.
            # if rank == 0 and (step + 1) % self.config.save_steps == 0:
            if rank == 0 and eva_iterator.should_save_checkpoint():
                print(f"Rank {rank} saving checkpoint at step {step}", flush=True)
                ckpt = {'step': step,
                        'recorder': recorder,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'train_dataset': train_dataset.where(),
                        'eval_dataset': eval_dataset.where()}

                if self.config.use_amp:
                    ckpt['amp'] = amp.state_dict()

                # torch.save(ckpt, self.config.save_checkpoint_path)
                temp_path="temp_checkpoint.pth"
                torch.save(ckpt, temp_path)

                # Atomically move the temporary file to the final destination
                os.replace(temp_path, self.config.save_checkpoint_path)

                # Because the checkpoint data allocates quite a lot of GPU
                # memories, we need to free the memories explicitly.
                del ckpt
                torch.cuda.empty_cache()
                print(f"Rank {rank} saved checkpoint at step {step}", flush=True)

                eva_iterator.save_checkpoint_done()
                print(f"Rank {rank} notified oteratpr that checkpoint is saved at step {step}", flush=True)
            
            if rank == 0:
                print(f"Rank {rank} finished step {step} at time {time.time()-start_time}", flush=True)
        
        if rank == 0:
            print(f"Rank {rank} training finished at time {time.time()-start_time}", flush=True)

        # Since the model is wrapped with `DistributedDataParallel` class in
        # distributed training environment, the original model can be accessed
        # by `module` attribute.
        if self.config.distributed:
            model = model.module

        # Save trained model weights and metrics recorded during the training.
        # if rank == 0:
        #     torch.save({'model': model.cpu().state_dict(),
        #                 'metrics': recorder.metrics},
        #                self.config.save_model_path)

    def _train_step(self,
                    rank: int,
                    dataset: Dataset,
                    model: nn.Module,
                    optimizer: optim.Optimizer,
                    scheduler: optim.lr_scheduler._LRScheduler
                    ) -> Dict[str, float]:
        model.train()
        optimizer.zero_grad()

        data = self._fetch_from(dataset, rank, self.config.batch_train)
        metrics = self.spec.train_objective(data, model)
        loss = metrics['loss']

        if self.config.use_amp:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        optimizer.step()
        scheduler.step()

        return {k: self._to_value(v) for k, v in metrics.items()}

    @torch.no_grad()
    def _eval_step(self, rank: int, dataset: Dataset, model: nn.Module
                   ) -> Dict[str, float]:
        model.eval()

        data = self._fetch_from(dataset, rank, self.config.batch_eval)
        metrics = self.spec.eval_objective(data, model)

        return {k: self._to_value(v) for k, v in metrics.items()}

    def _fetch_from(self, dataset: Dataset, rank: int, batch: int
                    ) -> Dict[str, torch.Tensor]:
        if self.config.distributed:
            # In distributed training environment, each process must ignore
            # sub-batches of other processes and fetch corresponding one only.
            batch = batch // self.config.gpus

            dataset.skip(rank * batch)
            data = dataset.fetch(batch)
            dataset.skip((self.config.gpus - rank - 1) * batch)
        else:
            data = dataset.fetch(self.config.batch_train)

        return {k: v.cuda() for k, v in data.items()}

    def _to_value(self, tensor: torch.Tensor) -> float:
        if self.config.distributed:
            tensor = tensor.clone()
            dist.all_reduce(tensor, op=dist.reduce_op.SUM)
            return (tensor / self.config.gpus).item()
        else:
            return tensor.item()
