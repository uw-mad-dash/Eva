# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: rpc/master.proto
# Protobuf Python Version: 4.25.0
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x10rpc/master.proto\"*\n\x15RegisterWorkerRequest\x12\x11\n\tworker_id\x18\x01 \x01(\x05\")\n\x16RegisterWorkerResponse\x12\x0f\n\x07success\x18\x01 \x01(\x08\")\n\x14SendHeartbeatRequest\x12\x11\n\tworker_id\x18\x01 \x01(\x05\"(\n\x15SendHeartbeatResponse\x12\x0f\n\x07success\x18\x01 \x01(\x08\";\n\x15TaskCompletionRequest\x12\x11\n\tworker_id\x18\x01 \x01(\x05\x12\x0f\n\x07task_id\x18\x02 \x01(\x05\")\n\x16TaskCompletionResponse\x12\x0f\n\x07success\x18\x01 \x01(\x08\x32\xd4\x01\n\x06Master\x12\x43\n\x0eRegisterWorker\x12\x16.RegisterWorkerRequest\x1a\x17.RegisterWorkerResponse\"\x00\x12@\n\rSendHeartbeat\x12\x15.SendHeartbeatRequest\x1a\x16.SendHeartbeatResponse\"\x00\x12\x43\n\x0eTaskCompletion\x12\x16.TaskCompletionRequest\x1a\x17.TaskCompletionResponse\"\x00\x62\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'rpc.master_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:
  DESCRIPTOR._options = None
  _globals['_REGISTERWORKERREQUEST']._serialized_start=20
  _globals['_REGISTERWORKERREQUEST']._serialized_end=62
  _globals['_REGISTERWORKERRESPONSE']._serialized_start=64
  _globals['_REGISTERWORKERRESPONSE']._serialized_end=105
  _globals['_SENDHEARTBEATREQUEST']._serialized_start=107
  _globals['_SENDHEARTBEATREQUEST']._serialized_end=148
  _globals['_SENDHEARTBEATRESPONSE']._serialized_start=150
  _globals['_SENDHEARTBEATRESPONSE']._serialized_end=190
  _globals['_TASKCOMPLETIONREQUEST']._serialized_start=192
  _globals['_TASKCOMPLETIONREQUEST']._serialized_end=251
  _globals['_TASKCOMPLETIONRESPONSE']._serialized_start=253
  _globals['_TASKCOMPLETIONRESPONSE']._serialized_end=294
  _globals['_MASTER']._serialized_start=297
  _globals['_MASTER']._serialized_end=509
# @@protoc_insertion_point(module_scope)
