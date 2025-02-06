# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

from rpc import iterator_pb2 as rpc_dot_iterator__pb2


class IteratorStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.GetThroughput = channel.unary_unary(
                '/Iterator/GetThroughput',
                request_serializer=rpc_dot_iterator__pb2.GetThroughputRequest.SerializeToString,
                response_deserializer=rpc_dot_iterator__pb2.GetThroughputResponse.FromString,
                )
        self.NotifySaveCheckpoint = channel.unary_unary(
                '/Iterator/NotifySaveCheckpoint',
                request_serializer=rpc_dot_iterator__pb2.NotifySaveCheckpointRequest.SerializeToString,
                response_deserializer=rpc_dot_iterator__pb2.NotifySaveCheckpointResponse.FromString,
                )


class IteratorServicer(object):
    """Missing associated documentation comment in .proto file."""

    def GetThroughput(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def NotifySaveCheckpoint(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_IteratorServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'GetThroughput': grpc.unary_unary_rpc_method_handler(
                    servicer.GetThroughput,
                    request_deserializer=rpc_dot_iterator__pb2.GetThroughputRequest.FromString,
                    response_serializer=rpc_dot_iterator__pb2.GetThroughputResponse.SerializeToString,
            ),
            'NotifySaveCheckpoint': grpc.unary_unary_rpc_method_handler(
                    servicer.NotifySaveCheckpoint,
                    request_deserializer=rpc_dot_iterator__pb2.NotifySaveCheckpointRequest.FromString,
                    response_serializer=rpc_dot_iterator__pb2.NotifySaveCheckpointResponse.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'Iterator', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class Iterator(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def GetThroughput(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/Iterator/GetThroughput',
            rpc_dot_iterator__pb2.GetThroughputRequest.SerializeToString,
            rpc_dot_iterator__pb2.GetThroughputResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def NotifySaveCheckpoint(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/Iterator/NotifySaveCheckpoint',
            rpc_dot_iterator__pb2.NotifySaveCheckpointRequest.SerializeToString,
            rpc_dot_iterator__pb2.NotifySaveCheckpointResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
