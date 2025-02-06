import boto3
import json
import logging
import subprocess
from .storage_manager import StorageManager

LOG_FORMAT = "{name}:{lineno}:{levelname} {message}"

class S3Manager(StorageManager):
    def __init__(self, bucket_name):
        self._s3 = boto3.client('s3')
        self._bucket_name = bucket_name
        self._logger = logging.getLogger("s3_manager")
        self._logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(LOG_FORMAT, style="{"))
        self._logger.addHandler(handler)
    
    @classmethod
    def from_config(cls, config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
            return cls(config["bucket_name"])

    def get_bucket_name(self):
        return self._bucket_name
    
    def get_dir(self, src_path, dst_path, exclude_list):
        """
        Download a directory from S3 to a local directory.
        src_path: the key of the S3 directory.
        dst_path: the local directory to download to.
        """
        self._logger.info(f"Downloading s3://{self._bucket_name}/{src_path} to {dst_path} with exclude list {exclude_list}")

        command = ['aws', 's3', 'sync', f's3://{self._bucket_name}/{src_path}', dst_path]
        for exclude in exclude_list:
            command.append(f'--exclude={exclude}')

        subprocess.run(command, stdout=subprocess.DEVNULL)

    
    def put_dir(self, src_path, dst_path):
        """
        Upload a directory to S3.
        src_path: the local directory to upload.
        dst_path: the key of the S3 directory.
        """
        self._logger.info(f"Uploading {src_path} to s3://{self._bucket_name}/{dst_path}")
        subprocess.run(['aws', 's3', 'sync', src_path, f's3://{self._bucket_name}/{dst_path}'], stdout=subprocess.DEVNULL)
    
    def remove_dir(self, path):
        """
        Remove a directory from S3.
        path: the key of the S3 directory.
        """
        self._logger.info(f"Removing s3://{self._bucket_name}/{path}")
        subprocess.run(['aws', 's3', 'rm', f's3://{self._bucket_name}/{path}', '--recursive'], stdout=subprocess.DEVNULL)
    
    def read_json(self, path):
        """
        Read a JSON file from S3.
        path: the key of the S3 file.
        """
        # self._logger.info(f"Reading s3://{self._bucket_name}/{path}")
        response = self._s3.get_object(Bucket=self._bucket_name, Key=path)
        return json.loads(response['Body'].read().decode('utf-8'))
    
    def dir_exists(self, path):
        """
        Check if a directory exists on S3.
        Checking logic: 
        * Folder should exists. 
        * Folder should not be empty.
        """
        if not path.endswith('/'):
            path = path+'/' 
        resp = self._s3.list_objects(Bucket=self._bucket_name, Prefix=path, Delimiter='/', MaxKeys=1)
        return 'Contents' in resp