import os
import time
from unittest.mock import patch
from torch.utils.data import DataLoader, Dataset

class SimpleDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def test_eva_iterator():
    # Set up environment variables
    os.environ["EVA_JOB_ID"] = "1"
    os.environ["EVA_TASK_ID"] = "1"
    os.environ["EVA_WORKER_IP_ADDR"] = "127.0.0.1"
    os.environ["EVA_WORKER_PORT"] = "50051"
    os.environ["EVA_ITERATOR_IP_ADDR"] = "127.0.0.1"
    os.environ["EVA_ITERATOR_PORT"] = "50052"

    # Mock the WorkerClient and serve function
    with patch("rpc.worker_client.WorkerClient") as mock_worker_client:
        mock_worker_client.return_value.GetStartTimestamp.return_value = (True, time.time())
        mock_worker_client.return_value.RegisterIterator.return_value = True
        mock_worker_client.return_value.DeregisterIterator.return_value = True

        from eva_iterator import EVAIterator

        # Mock the DataLoader
        data = list(range(10000000))
        data_loader = DataLoader(SimpleDataset(data), batch_size=32, shuffle=False)

        # Test the EVAIterator
        eva_iterator = EVAIterator(data_loader, min_sample_time=0, min_sample_steps=0)
        for i, batch in enumerate(eva_iterator):
            # do some computation
            # sum up the batch
            val = pow(sum(batch), 10)
            eva_iterator._get_throughput()
    
    print("Test passed")

if __name__ == "__main__":
    test_eva_iterator()