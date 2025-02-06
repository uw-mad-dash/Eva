import atexit
import logging
import os
import threading
import time
from collections import deque

from eva_iterator.rpc.worker_client import WorkerClient
from eva_iterator.rpc.iterator_server import serve

LOG_FORMAT = "{message}"

class IteratorAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        iterator = self.extra.get("iterator")
        timestamp = iterator.get_current_timestamp()
        timestamp = "{:.2f}".format(timestamp)
        absolute_timestamp = time.time()
        absolute_timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(absolute_timestamp))
        return "[{}] [{}] {}".format(absolute_timestamp, timestamp, msg), kwargs
    
    def removeHandler(self, handler):
        self.logger.removeHandler(handler)

class EVAIterator:
    # default log directory is current directory
    def __init__(self, data_loader, sample_duration=300, 
                 wait_for_save_checkpoint=False,
                 log_file="./eva_iterator.log", test_mode=False, test_mode_log_period=60):

        ##############################
        # Logging setup
        ##############################
        self._log_file = log_file
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)
        self._file_handler = logging.FileHandler(self._log_file)
        self._file_handler.setFormatter(logging.Formatter(LOG_FORMAT, style='{'))
        self._file_handler.setLevel(logging.DEBUG)
        logger.addHandler(self._file_handler)
        self._orig_logger = logger
        self._logger = IteratorAdapter(logger, {"iterator": self})
        self._logging_handler = self._logger

        atexit.register(self._close_file_handler)

        ##############################
        # member variables
        ##############################
        self._data_loader = data_loader
        self._sample_duration = sample_duration
        self._test_mode = test_mode
        self._test_mode_log_period = test_mode_log_period

        self._wait_for_save_checkpoint = wait_for_save_checkpoint
        self._should_save_checkpoint = False
        self._save_checkpoint_done = False
        self._save_checkpoint_done_cv = threading.Condition()

        # if test mode, set environment variables
        if self._test_mode:
            self._job_id = 0
            self._task_id = 0
            self._worker_ip_addr = None
            self._worker_port = None
            self._iterator_ip_addr = "localhost"
            self._iterator_port = 50123
            self._eva_start_timestamp = time.time()
        else:
            assert "EVA_JOB_ID" in os.environ, "EVA_JOB_ID not set"
            self._job_id = int(os.environ["EVA_JOB_ID"])
            self._task_id = int(os.environ["EVA_TASK_ID"])
            self._worker_ip_addr = os.environ["EVA_WORKER_IP_ADDR"]
            self._worker_port = int(os.environ["EVA_WORKER_PORT"])
            self._iterator_ip_addr = os.environ["EVA_ITERATOR_IP_ADDR"]
            self._iterator_port = int(os.environ["EVA_ITERATOR_PORT"])
            self._eva_start_timestamp = float(os.environ["EVA_START_TIMESTAMP"])

        if not self._test_mode:
            self._logger.info(f"Connecting to worker: {self._worker_ip_addr}:{self._worker_port}")
            self._worker_client = WorkerClient(self._worker_ip_addr, self._worker_port)
            self._worker_client.RegisterIterator(self._task_id)
            self._logger.info(f"Registered iterator with task id: {self._task_id}")

        callbacks = {
            "get_throughput": self._get_throughput,
            "notify_save_checkpoint": self._notify_save_checkpoint
        }
        self._server_thread = threading.Thread(
            target=serve,
            args=(self._iterator_ip_addr, self._iterator_port, callbacks)
        )
        self._server_thread.daemon = True
        self._server_thread.start()

        ##############################
        # Iterator init setup
        ##############################
        self._cur_step = 0
        self._first_step_time = None
        self._record_queue = deque() # contains (step, timestamp) tuples
        self._iterator = None # to be initialized in __iter__

        self._logger.info(f"Setup EVAIterator: {self}")

        ##############################
        # Test mode setup
        ##############################
        if self._test_mode:
            self._test_thread = threading.Thread(target=self.test_monitor)
            self._test_thread.daemon = True
            self._test_thread.start()

    def __str__(self):
        ret = "EvaIterator("
        for k, v in self.__dict__.items():
            ret += f"{k}={v}, "
        ret = ret[:-2] + ")"
        return ret
    
    ##############################
    # Public methods
    ##############################
    def get_current_timestamp(self):
        return time.time() - self._eva_start_timestamp
    
    def enable_cpu_affinity(self, loader_cores, compute_cores):
        return self._data_loader.enable_cpu_affinity(loader_cores, compute_cores)

    def should_save_checkpoint(self):
        """
        used by the application to check whether to save checkpoint
        """
        return self._should_save_checkpoint
    
    def save_checkpoint_done(self):
        """
        used by the application to notify that checkpoint is saved
        """
        with self._save_checkpoint_done_cv:
            self._logger.info(f"Save checkpoint done at step {self._cur_step}")
            self._should_save_checkpoint = False
            self._save_checkpoint_done_cv.notify_all()

    ##############################
    # Callbacks
    ##############################
    def _get_throughput(self):
        try:
            # remove old records
            cur_time = self.get_current_timestamp()
            self._logger.info(f"Getting throughput - cur_time: {cur_time}")

            if self._first_step_time is None:
                self._logger.info("Not ready -> First step time is None")
                return True, False, 0
            
            if cur_time - self._first_step_time < self._sample_duration:
                self._logger.info("Not ready -> Not enough time elapsed")
                return True, False, 0
            
            record_removed = []
            while len(self._record_queue) > 0 and cur_time - self._record_queue[0][1] > self._sample_duration:
                record = self._record_queue.popleft()
                record_removed.append(record)
            if len(record_removed) > 0:
                self._logger.info(f"Removed record: {record_removed}")
            
            if len(self._record_queue) == 0:
                self._logger.info("Not ready -> Record queue is empty")
                return True, False, 0
            
            # duration is the last record - first record
            duration = self._record_queue[-1][1] - self._record_queue[0][1]
            steps_completed = self._record_queue[-1][0] - self._record_queue[0][0]
            throughput = steps_completed / duration if duration > 0 else 0
            self._logger.info(f"Getting throughput - duration: {duration}, steps: {steps_completed}, throughput: {throughput}")
            
            return True, True, throughput
        except Exception as e:
            self._logger.error("Error getting throughput: {}".format(e))
            return False, False, 0
    
    def _notify_save_checkpoint(self):
        self._logger.info("Notifying save checkpoint")
        if self._wait_for_save_checkpoint:
            with self._save_checkpoint_done_cv:
                self._should_save_checkpoint = True
                self._logger.info("Waiting for save checkpoint")
                while self._should_save_checkpoint:
                    self._save_checkpoint_done_cv.wait()
                self._logger.info("Save checkpoint done")
        else:
            self._logger.info("Not waiting for save checkpoint")

        return True

    
    #############################
    # Private methods
    ##############################
    def _close_file_handler(self):
        self._logger.info("Closing file handler")
        self._logger.info("----------------------------------------------")
        self._logger.removeHandler(self._file_handler)
        self._file_handler.close()
    
    def __iter__(self):
        self._iterator = self._data_loader.__iter__()
        return self

    def __next__(self):
        try:
            data = next(self._iterator)
            cur_time = self.get_current_timestamp()
            if self._first_step_time is None:
                self._first_step_time = cur_time
            
            self._record_queue.append((self._cur_step, cur_time))
            self._cur_step += 1
            return data
        except StopIteration:
            self._logger.info("End of data")
            raise StopIteration
    
    def __len__(self):
        return len(self._data_loader)

    def test_monitor(self):
        while True:
            time.sleep(self._test_mode_log_period)
            self._get_throughput()
            self._notify_save_checkpoint()