import sys
from subprocess import call
import glob

GPU_ID = sys.argv[1]
version = sys.argv[2]

version_to_path = {"person_class":"person_class_only","all":"all_training_data","reasonable":"high_anchors"}


model_path = "/root/data/CalTech_models/{}/".format(version_to_path[version])
models = glob.glob(model_path+ "*.caffemodel")
#GPU_ID = "3"
net = "VGG16"
imdb = "caltech_{}".format(version)
script_path = "experiments/scripts/caltech_test_only.sh"

get_iterations = lambda model: int(model.split("_")[-1][:-11])
lower_bound = 50001
upper_bound = 70000

for model in models:
    iterations = get_iterations(model)
    if iterations > lower_bound and iterations < upper_bound:    
    	command = ["bash", script_path, GPU_ID, net, imdb, model]
    	print(" ".join(command))
    	call(command)
    
    
