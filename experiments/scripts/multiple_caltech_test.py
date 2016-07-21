import sys
from subprocess import call
import glob

GPU_ID = sys.argv[1]
version = sys.argv[2]

version_to_path = {"person_class":"person_class_only","all":"all_training_data","reasonable":"reasonable", "negative_reasonable":"reasonable"}


model_path = "/root/data/CalTech_models/{}/".format(version)
models = glob.glob(model_path+ "*.caffemodel")
#GPU_ID = "3"
net = "VGG16"
imdb = "caltech_{}".format(version_to_path[version])
script_path = "experiments/scripts/caltech_test_only.sh"


get_iterations = lambda model: int(model.split("_")[-1][:-11])
lower_bound = 80000
upper_bound = float("inf")
#target = 110000
for model in models:
    
    command = ["bash", script_path, GPU_ID, net, imdb, model]
    iterations = get_iterations(model)
    if iterations > lower_bound and iterations < upper_bound:
    #if iterations == target: 	
        print("Starting Evaluating iterations: {}".format(iterations))
        call(command)
    
    
