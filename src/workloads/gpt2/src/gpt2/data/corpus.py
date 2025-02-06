import torch
from gpt2.data import Dataset, Vocab
from typing import Dict, Any, List, Optional

class TokenizedCorpus(Dataset):
    def __init__(self,
                 corpus_path: str,
                 vocab: Vocab,
                 seq_len: int,
                 repeat: bool = True):
        with open(corpus_path, 'r', encoding='utf-8') as corpus_file:
            self.data = []
            for line in corpus_file:
                indices = [vocab[t] for t in line.split()]
                if len(indices) + 2 <= seq_len:  # Filter sequences longer than seq_len
                    indices = [vocab.bos_idx] + indices + [vocab.eos_idx]
                    indices += [vocab.pad_idx] * (seq_len - len(indices) + 1)
                    self.data.append({'input': indices[:-1], 'output': indices[1:]})
        print("finish processing corpus")
        self.vocab = vocab
        self.seq_len = seq_len
        self.repeat = repeat
        self.pointer = 0

    def skip(self, count: int):
        self.pointer = (self.pointer + count) % len(self.data)

    def fetch(self, batch: Optional[int] = None) -> Dict[str, torch.Tensor]:
        if batch is None:
            data = self.data[self.pointer]
            self.pointer = (self.pointer + 1) % len(self.data)
        else:
            data = self.data[self.pointer:self.pointer+batch]
            self.pointer = (self.pointer + batch) % len(self.data)
        return {k: torch.tensor([d[k] for d in data], dtype=torch.long) for k in data[0]}

    def where(self) -> Dict[str, Any]:
        return {'offset': self.pointer}

    def assign(self, where: Dict[str, Any]):
        self.pointer = where['offset']

