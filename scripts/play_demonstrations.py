"""
        Render and save screenshots from saved demonstrations 
        
        Inputs: 
            .../environment_dir
            .../demos_dir
            .../images_dir
        
        Outputs: 
            screenshots saved in .../images_dir/001.png, 002.png ... (the image name is the frame number)

"""

import os
import sys
import torch
from context import needlemaster as nm
from pdb import set_trace as woah
from play import playback

environment_dir    = '/home/molly/workspace/Surgical_Automation/experiments/needle_master_tools/environments/'
demonstration_dir  = '/home/molly/workspace/Surgical_Automation/experiments/needle_master_tools/demonstrations/'
images_path        = '/home/molly/workspace/Surgical_Automation/experiments/needle_master_tools/images/'

demo_list = os.listdir(demonstration_dir)
for i in range(54, len(demo_list)):
    demo = demo_list[i]
    print("*********************************")
    print(str(i) + "\t Rendering " + demo)
    demo_parts = demo.split('_')
    env_level = demo_parts[1]
    
    demo_path = demonstration_dir + demo
    env_path = environment_dir + 'environment_' + env_level + '.txt'
    
    playback(env_path, demo_path, images_path)
    