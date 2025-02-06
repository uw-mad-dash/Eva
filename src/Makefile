rpc_stubs:
	python -m grpc_tools.protoc -Irpc=rpc/protobuf --python_out=. --grpc_python_out=. rpc/protobuf/master.proto
	python -m grpc_tools.protoc -Irpc=rpc/protobuf --python_out=. --grpc_python_out=. rpc/protobuf/worker.proto
	python -m grpc_tools.protoc -Irpc=rpc/protobuf --python_out=. --grpc_python_out=. rpc/protobuf/submission_manager.proto
	python -m grpc_tools.protoc -Irpc=rpc/protobuf --python_out=. --grpc_python_out=. rpc/protobuf/simulator.proto
	python -m grpc_tools.protoc -Irpc=rpc/protobuf --python_out=. --grpc_python_out=. rpc/protobuf/simulation_event_receiver.proto
	python -m grpc_tools.protoc -Irpc=rpc/protobuf --python_out=. --grpc_python_out=. rpc/protobuf/iterator.proto

	python -m grpc_tools.protoc -Ieva_iterator/rpc=rpc/protobuf --python_out=. --grpc_python_out=. rpc/protobuf/iterator.proto
	python -m grpc_tools.protoc -Ieva_iterator/rpc=rpc/protobuf --python_out=. --grpc_python_out=. rpc/protobuf/worker.proto
clean:
	rm -rf rpc/*_pb2.py rpc/*_pb2_grpc.py