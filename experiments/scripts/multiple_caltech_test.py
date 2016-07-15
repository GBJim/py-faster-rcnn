import sys
from subprocess import call
import glob

version = sys.argv[1]

version_to_path = {"person_class":"person_class_only","all":"all_training_data","reasonable":"reasonable"}


model_path = "/root/data/CalTech_models/{}/".format(version_to_path[version])
models = glob.glob(model_path+ "*.caffemodel")
GPU_ID = "3"
net = "VGG16"
imdb = "caltech_{}".format(version)
script_path = "experiments/scripts/caltech_test_only.sh"


for model in models:
    
    command = ["bash", script_path, GPU_ID, net, imdb, model]
    print(" ".join(command))
    call(command)
    
    
