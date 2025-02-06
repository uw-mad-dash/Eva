import grpc
import logging

import eva_iterator.rpc.iterator_pb2_grpc as iterator_pb2_grpc
import eva_iterator.rpc.iterator_pb2 as iterator_pb2

from concurrent import futures

LOG_FORMAT = '{name}:{lineno}:{levelname} [{asctime}] {message}'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

class IteratorServer(iterator_pb2_grpc.IteratorServicer):
    def __init__(self, callbacks, logger):
        self._callbacks = callbacks
        self._logger = logger
    
    def GetThroughput(self, request, context):
        self._logger.info("Getting throughput")
        success, ready, throughput = self._callbacks["get_throughput"]()
        return iterator_pb2.GetThroughputResponse(success=success, ready=ready, throughput=throughput)
    
    def NotifySaveCheckpoint(self, request, context):
        self._logger.info("Notifying save checkpoint")
        success = self._callbacks["notify_save_checkpoint"]()
        return iterator_pb2.NotifySaveCheckpointResponse(success=success)

def serve(ip_addr, port, callbacks):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT, style='{'))
    logger.addHandler(handler)

    server = grpc.server(futures.ThreadPoolExecutor())
    iterator_pb2_grpc.add_IteratorServicer_to_server(
        IteratorServer(callbacks, logger), server)
    server.add_insecure_port('%s:%d' % (ip_addr, port))
    server.start()
    server.wait_for_termination()