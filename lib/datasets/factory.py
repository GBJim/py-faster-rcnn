# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""

__sets = {}

from datasets.caltech import caltech
from datasets.pascal_voc import pascal_voc
from datasets.coco import coco
import numpy as np


#Set up Caltech
    
for version in ["all", "reasonable", "person_class"]:
    for split in ['train', 'val', 'trainval', 'test']:         #I need an all in here meaning that I will conain both data1
        name = 'caltech_{}_{}'.format(version, split)
        __sets[name] = (lambda split=split, version=version : caltech(version, split))

# Set up voc_<year>_<split> using selective search "fast" mode
for year in ['2007', '2012',"0712"]:        #I need an (2007+12)  in here
    for split in ['train', 'val', 'trainval', 'test']:         #I need an all in here meaning that I will conain both data1
        name = 'voc_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: pascal_voc(split, year))

# Set up coco_2014_<split>
for year in ['2014']:
    for split in ['train', 'val', 'minival', 'valminusminival']:
        name = 'coco_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: coco(split, year))

# Set up coco_2015_<split>
for year in ['2015']:
    for split in ['test', 'test-dev']:
        name = 'coco_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: coco(split, year))

def get_imdb(name):
    """Get an imdb (image database) by name."""
    print(__sets)
    if not __sets.has_key(name):
        raise KeyError('Unknown dataset: {}'.format(name))
    return __sets[name]()

def list_imdbs():
    """List all registered imdbs."""
    return __sets.keys()
