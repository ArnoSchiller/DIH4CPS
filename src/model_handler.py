import os
import glob
import boto3 

from configuration import global_cloud_access_key, global_cloud_secret_key, global_cloud_url, global_trained_model_bucket, global_created_model_bucket 

class ModelHandler:

    model_label_map_name    = "model_label_map.pbtxt"
    model_graph_name        = "created_model_graph"
    test_image_name         = "test_image.png"

    base_dir_path           = os.path.dirname(__file__)
    label_map_path          = os.path.join(base_dir_path, model_label_map_name)
    model_graph_path        =  os.path.join(base_dir_path, model_graph_name)

    cloud_access_key        = global_cloud_access_key
    cloud_secret_key        = global_cloud_secret_key
    cloud_url               = global_cloud_url

    trained_model_bucket    = global_trained_model_bucket
    created_model_bucket    = global_created_model_bucket
    model_names             = ["tf_API_data1_v01", "tf_API_data2_v01"]

    def __init__(self):
        self.s3_client = boto3.client('s3',
                        aws_access_key_id = self.cloud_access_key,
                        aws_secret_access_key = self.cloud_secret_key,
                        endpoint_url=self.cloud_url, 
                        verify=True,
                        config=boto3.session.Config(signature_version='s3v4'))

        self.s3_resource = boto3.resource('s3',
                        aws_access_key_id = self.cloud_access_key,
                        aws_secret_access_key = self.cloud_secret_key,
                        endpoint_url=self.cloud_url, 
                        verify=True,
                        config=boto3.session.Config(signature_version='s3v4'))

        self.model_names = self.get_model_names()

    def download_test_image(self, base_path=None):

        if base_path is None:
            base_path = self.base_dir_path

        bucket = self.s3_resource.Bucket(self.trained_model_bucket)        
        for bucket_object in bucket.objects.all():
            object_name = str(bucket_object.key)
            if object_name.count(self.test_image_name) > 0:

                file_path = os.path.join(base_path, self.test_image_name)
                if os.path.exists(file_path):
                    os.remove(file_path)

                self.s3_client.download_file(self.trained_model_bucket, object_name, file_path)
        return os.path.exists(file_path)

    def download_trained_model(self, model_name=None, base_path=None):
        
        if model_name is None:
            model_name = self.get_latest_model_name()

        if not self.model_names.count(model_name) > 0:
            print("model name is not available.")
            return
        
        if base_path is None:
            base_path = self.base_dir_path

        if not os.path.exists(base_path):
            print("base path is not available.")
            return
        
        lm_res = self.download_label_map(model_name, base_path)
        mg_res = self.download_model_graph(model_name, base_path)

        # download failed
        if not (lm_res and mg_res):
            self.remove_model_files()
            return False

        return True
    
    def download_utils(self, base_path=None, with_visualisation=False):
        """
        Download utils files from minio cloud and verify the download. If the download was not successfull, 
        """
        if base_path is None:
            base_path = self.base_dir_path

        utils_path = os.path.join(base_path, "utils")

        if not os.path.exists(utils_path):
            os.mkdir(utils_path)
        
        download_files = []

        bucket = self.s3_resource.Bucket(self.trained_model_bucket)        
        for bucket_object in bucket.objects.all():
            object_name = str(bucket_object.key)
            # download utils files
            if object_name.count("utils") > 0:

                # do not download visualization tools if not needed
                if not with_visualisation:
                    if object_name.count("visualization") > 0:
                        continue

                file_name = object_name.split("/")[-1]
                file_path = os.path.join(utils_path, file_name)
                download_files.append(file_path)
                
                if os.path.exists(file_path):
                    os.remove(file_path)

                self.s3_client.download_file(self.trained_model_bucket, object_name, file_path)
        
        return self.verify_download(download_files)
        
        
    def download_label_map(self, model_name, base_path):

        label_map_path = base_path
        
        if not os.path.exists(label_map_path):
            os.mkdir(label_map_path)

        label_map_name = "object-detection.pbtxt"

        download_files = []
        bucket = self.s3_resource.Bucket(self.trained_model_bucket) 
        for bucket_object in bucket.objects.all():
            object_name = str(bucket_object.key)
            # download utils files
            if object_name.count(model_name) > 0 and object_name.count(label_map_name) > 0:
                
                download_files.append(self.label_map_path)
                        
                self.s3_client.download_file(self.trained_model_bucket, object_name, self.label_map_path)

        return self.verify_download(download_files)
        
        
         
    def download_model_graph(self, model_name, base_path, model_graph_name=None):
        
        possible_graphs = self.get_model_graph_names(model_name)

        if model_graph_name is None:
            model_graph_name = self.get_latest_graph(model_name)

        if not possible_graphs.count(model_graph_name) > 0:
            print("Graph is not available")
            return False

        model_graph_path = os.path.join(base_path, "created_model_graph")
        if not os.path.exists(model_graph_path):
            os.mkdir(model_graph_path)

        saved_model_path = os.path.join(model_graph_path, "saved_model")
        if not os.path.exists(saved_model_path):
            os.mkdir(saved_model_path)

        download_files = []
        bucket = self.s3_resource.Bucket(self.trained_model_bucket) 
        for bucket_object in bucket.objects.all():
            object_name = str(bucket_object.key)
            # download utils files
            if object_name.count(model_graph_name) > 0:
                file_name = object_name.split("/")[-1]
                if object_name.count("saved_model") > 0:
                    file_path = os.path.join(saved_model_path, file_name)
                else:
                    file_path = os.path.join(model_graph_path, file_name)
                download_files.append(file_path)
                        
                self.s3_client.download_file(self.trained_model_bucket, object_name, file_path)

        # verify download
        download_complete = self.verify_download(download_files)
        
        if not download_complete:
            for downloaded_file in glob.glob(os.path.join(saved_model_path,"*")):
                os.remove(downloaded_file)
            os.rmdir(saved_model_path)
            for downloaded_file in glob.glob(os.path.join(model_graph_path,"*")):
                os.remove(downloaded_file)
            os.rmdir(model_graph_path)
            return False
        return True

    def get_latest_graph(self, model_name):
        """
        Returns the model graph name with a specific name and the latest version tag.
        """
        graphs = self.get_model_graph_names(model_name)

        latest_version = 0
        latest_graph_name = ""

        for graph in graphs:
            version = int(graph.split("_")[-1])
            if version > latest_version:
                latest_version = version
                latest_graph_name = graph
        
        return latest_graph_name

    def get_model_graph_names(self, model_name):
        """
        Get every model graph name uploaded to the minio cloud.
        """
        graphs = []
        graph_prefix = model_name + "_graph_"
        bucket = self.s3_resource.Bucket(self.trained_model_bucket) 

        for bucket_object in bucket.objects.all():
            object_name = str(bucket_object.key)
            if object_name.count(graph_prefix) > 0:
                graph_name = object_name.split("/")[0]
                if not graphs.count(graph_name) > 0:
                    graphs.append(graph_name)
        return graphs
    
    def get_model_names(self):
        """
        Returns a list of models saved in the minio cloud. 
        """
        models = []
        bucket = self.s3_resource.Bucket(self.trained_model_bucket) 
        for bucket_object in bucket.objects.all():
            object_name = str(bucket_object.key)
            folder_name = object_name.split("/")[0]
            if folder_name.count("data") > 0:
                tags = folder_name.split("data")[1]
                if tags.count("_") == 1:
                    models.append(folder_name)
        return models

    def get_latest_model_name(self, model_type="tf_API_data1"):
        
        latest_version = 0
        latest_model = None

        models = self.get_model_names()
        for name in models:
            if name.count(model_type) > 0:
                version_tag = name.split("_")[-1].split("v")[1]
                print(version_tag)
                if int(version_tag) > latest_version:
                    latest_model = name
        return latest_model

    def remove_model_files(self):
        paths = []
        if os.path.exists(os.path.join(self.model_graph_path, "saved_model")):
            paths.append(os.path.join(self.model_graph_path, "saved_model"))
        if os.path.exists(self.model_graph_path):
            paths.append(self.model_graph_path)
        
        for path in paths:
            for file_path in glob.glob(os.path.join(path,"*")):
                os.remove(file_path)
            os.rmdir(path)

        if os.path.exists(self.label_map_path):
            os.remove(self.label_map_path)

    def validate_model(self):
        if not os.path.exists(self.label_map_path):
            return False 
        

        return True

    def verify_download(self, download_paths):
        """
        Proof if every download path exists. Return false if there is a file that was not downloaded correctly. 
        """
        for path in download_paths:
            if not os.path.exists(path):
                print("Downloading file {} failed. Restore download.".format(path))
                return False
        return True

"""
mh = ModelHandler()
#mh.get_latest_graph("tf_API_data1_v01")
# mh.get_model_graph_names("tf_API_data1_v01")
mh.download_trained_model(mh.model_names[1], os.path.dirname(__file__))
"""