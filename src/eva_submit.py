import argparse
from rpc.submission_manager_client import SubmissionManagerClient

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eva-ip-addr", type=str, default="localhost")
    parser.add_argument("--eva-port", type=int, default=50422)
    parser.add_argument("--local-working-dir", type=str)
    parser.add_argument("--global-working-dir", type=str)
    # TODO: an option to send a tarball to master

    args = parser.parse_args()

    client = SubmissionManagerClient(args.eva_ip_addr, args.eva_port)

    if args.local_working_dir:
        # ask master for storage manager info
        response = client.GetStorageManagerConfig()
        mod = __import__("storage_manager", fromlist=[response.class_name])
        storage_manager_class = getattr(mod, response.class_name)
        storage_manager = storage_manager_class(**response.args)
        # check if the directory exists on cloud
        if storage_manager.dir_exists(args.global_working_dir):
            print("The directory already exists on cloud. Maybe change the dir name?")
            return
        storage_manager.put_dir(args.local_working_dir, args.global_working_dir)


    response = client.Submit(args.global_working_dir)
    if response.success:
        print(f"Job id: {response.job_id}")
    else:
        print("failed")

if __name__ == "__main__":
    main()