from subprocess import call
import glob
model_path = "/root/data/CalTech_models/person_class_only/"
models = glob.glob(model_path+ "*.caffemodel")
GPU_ID = "2"
net = "VGG16"
imdb = "caltech"
script_path = "experiments/scripts/caltech_test_only.sh"


for model in models:
    command = ["bash", script_path, GPU_ID, net, imdb, model]
    
    call(command)
    break