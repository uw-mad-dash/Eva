import boto3
from botocore.config import Config
import shlex
import time

def launch_instances(launch_cfg, instance_name=None):
    client = boto3.client("ec2", region_name=launch_cfg["region"])
    ec2 = boto3.resource("ec2", region_name=launch_cfg["region"])
    if instance_name is None:
        instance_name = f"eva_{launch_cfg['instance_type']}"

    # try availability zone if receive error
    az_list = ["a", "b", "c", "d", "e", "f"]
    response = None
    instance_found = False
    while not instance_found:
        for az in az_list:
            try:
                response = client.run_instances(
                    MaxCount=launch_cfg["instance_count"],
                    MinCount=launch_cfg["instance_count"],
                    ImageId=launch_cfg["ami_id"],
                    InstanceType=launch_cfg["instance_type"],
                    KeyName=launch_cfg["key_name"],
                    EbsOptimized=True,
                    IamInstanceProfile={"Name": launch_cfg["iam_role_name"]},
                    Placement={"AvailabilityZone": launch_cfg["region"] + az},
                    SecurityGroups=launch_cfg["security_group"],
                    BlockDeviceMappings=[
                        {
                            "DeviceName": "/dev/sda1",
                            "Ebs": {
                                "DeleteOnTermination": True,
                                "VolumeSize": launch_cfg.get("volume_size", 768),
                                "VolumeType": launch_cfg.get("volume_type", "gp2"),
                            },
                        }
                    ],
                    TagSpecifications=[
                        {
                            "ResourceType": "instance",
                            "Tags": [
                                {"Key": "Name", "Value": instance_name},
                            ],
                        }
                    ]
                )
                instance_found = True
                break
            except Exception as e:
                print(f"Error: {e}", flush=True)
                continue
    # response = client.run_instances(
    #     MaxCount=launch_cfg["instance_count"],
    #     MinCount=launch_cfg["instance_count"],
    #     ImageId=launch_cfg["ami_id"],
    #     InstanceType=launch_cfg["instance_type"],
    #     KeyName=launch_cfg["key_name"],
    #     EbsOptimized=True,
    #     IamInstanceProfile={"Name": launch_cfg["iam_role_name"]},
    #     Placement={"GroupName": launch_cfg["group_name"], "AvailabilityZone": launch_cfg["region"] + "f"},
    #     # Placement={"GroupName": launch_cfg["group_name"]},
    #     SecurityGroups=launch_cfg["security_group"],
    # )

    instance_ids = list()

    for request in response["Instances"]:
        instance_ids.append(request["InstanceId"])
    time.sleep(5)
    loop = True
    while loop:
        loop = False
        print("Instance ids {}".format(instance_ids))
        response = client.describe_instance_status(
            InstanceIds=instance_ids, IncludeAllInstances=True
        )
        # print("Response {}".format(response))
        for status in response["InstanceStatuses"]:
            print("Status {}".format(status["InstanceState"]["Name"]))
            if status["InstanceState"]["Name"] != "running":
                loop = True
                time.sleep(5)
    print("All instances are running ...")

    instance_collection = ec2.instances.filter(
        Filters=[{"Name": "instance-id", "Values": instance_ids}]
    )
    print("Instance collection {}".format(instance_collection))
    private_ips = []
    public_ips = []
    for instance in instance_collection:
        print(instance.private_ip_address)
        private_ips.append(instance.private_ip_address)
        print(instance.public_ip_address)
        public_ips.append(instance.public_ip_address)
    return (public_ips, private_ips, instance_ids)

def terminate_instances(instance_ids, region):
    client = boto3.client("ec2", region_name=region)
    # TODO: change to terminate
    response = client.stop_instances(InstanceIds=instance_ids)
    for request in response["StoppingInstances"]:
        print(f"Instance {request['InstanceId']} with status {request['CurrentState']['Name']}")

    # response = client.terminate_instances(InstanceIds=instance_ids)
    # for request in response["TerminatingInstances"]:
    #     print(f"Instance {request['InstanceId']} with status {request['CurrentState']['Name']}")

def execute_commands_on_instance(instance_id, region, commands, blocking=True):
    """Runs commands on remote linux instance
    :param commands: a list of strings, each one a command to execute on the instances
    :param instance_id: instance_id string, of the instance on which to execute the command
    :return: the response from the send_command function (check the boto3 docs for ssm client.send_command() )
    """
    config = Config(retries={"max_attempts": 10, "mode": "standard"})
    client = boto3.client("ssm", region_name=region, config=config)

    # check if the instance is ready to receive commands
    while client.get_connection_status(Target=instance_id)["Status"] != "connected":
        time.sleep(1)

    # print("Instance is ready to receive commands", flush=True)
    resp = client.send_command(
        DocumentName="AWS-RunShellScript", # One of AWS' preconfigured documents
        Parameters={'commands': commands},
        InstanceIds=[instance_id],
    )

    command_id = resp['Command']['CommandId']
    time.sleep(5)

    success = True
    output = ''
    while True and blocking:
        try:
            result = client.get_command_invocation(
                CommandId=command_id,
                InstanceId=instance_id
            )
        except Exception as e:
            # retry if InvocationDoesNotExist error
            print(f"Error: {e}", flush=True)
            print("Retrying...", flush=True)
            time.sleep(5)
            continue
        if result['Status'] == 'Success':
            success = True
            output = result['StandardOutputContent']
            break
        elif result['Status'] == 'Failed':
            success = False
            output = result['StandardErrorContent']
            break
        else:
            output += result['StandardOutputContent']

    return success, output

if __name__ == "__main__":
    worker_config_path = "~/eva_worker/eva_worker_config.json"
    worker_config = {
        "id": 0,
        "ip_addr": "blahblah"
    }
    import os
    import json
    json_string = json.dumps(worker_config).replace('"', '\\"')
    commands = [
        f'runuser -l ubuntu -c "mkdir -p {os.path.dirname(worker_config_path)}"',
        f'runuser -l ubuntu -c "echo \'{json_string}\' > {worker_config_path}"'
        # f'runuser -l ubuntu -c "cd ~/eva/src && python3 eva_worker.py --config-path {worker_config_path}"'
    ]
    print(commands[1], flush=True)

    success, output = execute_commands_on_instance('i-04c79f1b52855148c', "us-east-1", [commands[1]])
    
    print(f"Success: {success}, Output: {output}", flush=True)