# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: rpc/simulator.proto
# Protobuf Python Version: 4.25.0
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x13rpc/simulator.proto\"I\n\x1cRegisterEventReceiverRequest\x12\n\n\x02id\x18\x01 \x01(\t\x12\x0f\n\x07ip_addr\x18\x02 \x01(\t\x12\x0c\n\x04port\x18\x03 \x01(\x05\"0\n\x1dRegisterEventReceiverResponse\x12\x0f\n\x07success\x18\x01 \x01(\x08\"\x15\n\x13GetTimeStampRequest\")\n\x14GetTimeStampResponse\x12\x11\n\ttimestamp\x18\x01 \x01(\t\"\x18\n\x16GetNewCommandIdRequest\"-\n\x17GetNewCommandIdResponse\x12\x12\n\ncommand_id\x18\x01 \x01(\x05\x32\xec\x01\n\tSimulator\x12X\n\x15RegisterEventReceiver\x12\x1d.RegisterEventReceiverRequest\x1a\x1e.RegisterEventReceiverResponse\"\x00\x12=\n\x0cGetTimeStamp\x12\x14.GetTimeStampRequest\x1a\x15.GetTimeStampResponse\"\x00\x12\x46\n\x0fGetNewCommandId\x12\x17.GetNewCommandIdRequest\x1a\x18.GetNewCommandIdResponse\"\x00\x62\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'rpc.simulator_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:
  DESCRIPTOR._options = None
  _globals['_REGISTEREVENTRECEIVERREQUEST']._serialized_start=23
  _globals['_REGISTEREVENTRECEIVERREQUEST']._serialized_end=96
  _globals['_REGISTEREVENTRECEIVERRESPONSE']._serialized_start=98
  _globals['_REGISTEREVENTRECEIVERRESPONSE']._serialized_end=146
  _globals['_GETTIMESTAMPREQUEST']._serialized_start=148
  _globals['_GETTIMESTAMPREQUEST']._serialized_end=169
  _globals['_GETTIMESTAMPRESPONSE']._serialized_start=171
  _globals['_GETTIMESTAMPRESPONSE']._serialized_end=212
  _globals['_GETNEWCOMMANDIDREQUEST']._serialized_start=214
  _globals['_GETNEWCOMMANDIDREQUEST']._serialized_end=238
  _globals['_GETNEWCOMMANDIDRESPONSE']._serialized_start=240
  _globals['_GETNEWCOMMANDIDRESPONSE']._serialized_end=285
  _globals['_SIMULATOR']._serialized_start=288
  _globals['_SIMULATOR']._serialized_end=524
# @@protoc_insertion_point(module_scope)
