from __future__ import print_function, division
import os
os.environ["OMP_NUM_THREADS"] = "1"
import argparse
import torch
import torch.multiprocessing as mp
from environment import atari_env
from utils import read_config
from model import A3Clstm
from train import train
from test import test
from shared_optim import SharedRMSprop, SharedAdam
#from gym.configuration import undo_logger_setup
import time
import dill
from eva_iterator import EVAIterator
EVA_SAMPLE_DURATION = 30
EVA_ITERATOR_TEST_MODE = False
EVA_ITERATOR_TEST_MODE_LOG_PERIOD = 60

#undo_logger_setup()"
parser = argparse.ArgumentParser(description="A3C")
parser.add_argument(
    "-l", "--lr", type=float, default=0.0001, help="learning rate (default: 0.0001)"
)
parser.add_argument(
    "-ec",
    "--entropy-coef",
    type=float,
    default=0.01,
    help="entropy loss coefficient (default: 0.01)",
)
parser.add_argument(
    "-vc",
    "--value-coef",
    type=float,
    default=0.5,
    help="value coefficient (default: 0.5)",
)
parser.add_argument(
    "-g",
    "--gamma",
    type=float,
    default=0.99,
    help="discount factor for rewards (default: 0.99)",
)
parser.add_argument(
    "-t", "--tau", type=float, default=1.00, help="parameter for GAE (default: 1.00)"
)
parser.add_argument(
    "-s", "--seed", type=int, default=1, help="random seed (default: 1)"
)
parser.add_argument(
    "-w",
    "--workers",
    type=int,
    default=2,
    help="how many training processes to use (default: 32)",
)
parser.add_argument(
    "-ns",
    "--num-steps",
    type=int,
    default=20,
    help="number of forward steps in A3C (default: 20)",
)
parser.add_argument(
    "-mel",
    "--max-episode-length",
    type=int,
    default=10000,
    help="maximum length of an episode (default: 10000)",
)
parser.add_argument(
    "-ev",
    "--env",
    default="PongNoFrameSkip-v4",
    help="environment to train on (default: PongNoFrameSkip-v4)",
)
parser.add_argument(
    "-so",
    "--shared-optimizer",
    default=True,
    help="use an optimizer with shared statistics.",
)
parser.add_argument("-ld", "--load", action="store_true", help="load a trained model")
parser.add_argument(
    "-sm",
    "--save-max",
    action="store_true",
    help="Save model on every test run high score matched or bested",
)
parser.add_argument(
    "-o",
    "--optimizer",
    default="Adam",
    choices=["Adam", "RMSprop"],
    help="optimizer choice of Adam or RMSprop",
)
parser.add_argument(
    "-lmd",
    "--load-model-dir",
    default="trained_models/",
    help="folder to load trained models from",
)
parser.add_argument(
    "-smd",
    "--save-model-dir",
    default="trained_models/",
    help="folder to save trained models",
)
parser.add_argument("-lg", "--log-dir", default="logs/", help="folder to save logs")
parser.add_argument(
    "-gp",
    "--gpu-ids",
    type=int,
    default=[-1],
    nargs="+",
    help="GPUs to use [-1 CPU only] (default: -1)",
)
parser.add_argument(
    "-a", "--amsgrad", action="store_true", help="Adam optimizer amsgrad parameter"
)
parser.add_argument(
    "--skip-rate",
    type=int,
    default=4,
    metavar='SR',
    help="frame skip rate (default: 4)")
parser.add_argument(
    "-hs",
    "--hidden-size",
    type=int,
    default=512,
    help="LSTM Cell number of features in the hidden state h",
)
parser.add_argument(
    "-tl",
    "--tensorboard-logger",
    action="store_true",
    help="Creates tensorboard logger to see graph of model, view model weights and biases, and monitor test agent reward progress",
)
parser.add_argument(
    "-evc", "--env-config",
    default="a3c_config.json",
    help="environment to crop and resize info (default: a3c_config.json)")
parser.add_argument(
    "-dss",
    "--distributed-step-size",
    type=int,
    default=[],
    nargs="+",
    help="use different step size among workers by using a list of step sizes to distributed among workers to use (default: [])",
)
parser.add_argument(
    "-episodes",
    "--episodes",
    type=int,
    default=3000,
    help="number of episodes to train for",
)

# Based on
# https://github.com/pytorch/examples/tree/master/mnist_hogwild
# Training settings
# Implemented multiprocessing using locks but was not beneficial. Hogwild
# training was far superior

def monitor(num_episodes_trained, args):
    iterator = EVAIterator(
        range(num_episodes_trained.item(), args.episodes), 
        sample_duration=EVA_SAMPLE_DURATION,
        test_mode=EVA_ITERATOR_TEST_MODE,
        test_mode_log_period=EVA_ITERATOR_TEST_MODE_LOG_PERIOD
    )
    for i in iterator:
        while i >= num_episodes_trained:
            time.sleep(1)


if __name__ == '__main__':
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    if args.gpu_ids != [-1]:
        torch.cuda.manual_seed(args.seed)
        mp.set_start_method("spawn")
    setup_json = read_config(args.env_config)
    env_conf = setup_json["Default"]
    for i in setup_json.keys():
        if i in args.env:
            env_conf = setup_json[i]
    env = atari_env(args.env, env_conf, args)
    shared_model = A3Clstm(env.observation_space.shape[0], env.action_space, args)
    num_episodes_trained = torch.tensor(0)
    if args.load:
        # see if there is a trained model to load
        if os.path.isfile(f"{args.load_model_dir}{args.env}_last.dat"):
            print(f"Loading model from {args.load_model_dir}{args.env}_last.dat")
            saved_state = torch.load(
                f"{args.load_model_dir}{args.env}_last.dat",
                map_location=lambda storage, loc: storage,
            )
            num_episodes_trained = torch.tensor(
                saved_state["num_episodes_trained"]
            )
            del saved_state["num_episodes_trained"]
            shared_model.load_state_dict(saved_state)
        else:
            print("No model found")
    shared_model.share_memory()
    num_episodes_trained.share_memory_()
    print(f"Number of episodes trained: {num_episodes_trained.item()}")
    print(f"Number of epochs {args.episodes}")

    if args.shared_optimizer:
        if args.optimizer == 'RMSprop':
            optimizer = SharedRMSprop(shared_model.parameters(), lr=args.lr)
        if args.optimizer == 'Adam':
            optimizer = SharedAdam(
                shared_model.parameters(), lr=args.lr, amsgrad=args.amsgrad)
        optimizer.share_memory()
    else:
        optimizer = None

    processes = []

    p = mp.Process(target=monitor, args=(num_episodes_trained, args))
    p.start()
    processes.append(p)
    time.sleep(0.001)

    p = mp.Process(target=test, args=(args, shared_model, num_episodes_trained, env_conf))
    p.start()
    processes.append(p)
    time.sleep(0.001)
    for rank in range(0, args.workers):
        p = mp.Process(
            target=train, args=(rank, args, shared_model, optimizer, num_episodes_trained, env_conf))
        p.start()
        processes.append(p)
        time.sleep(0.001)
    for p in processes:
        time.sleep(0.001)
        p.join()
