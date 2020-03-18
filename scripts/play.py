import os
import sys
import torch
from context import needlemaster as nm
from pdb import set_trace as woah

def playback(env_path, demo_path, im_path):
    """
            Molly 11/30/2018

            read in an environment and demonstration and "hallucinate" the
            screen images

            Args:
                env_path: path/to/corresponding/environment/file
                demo_path: path/to/demo/file
    """
    environment = nm.Environment(env_path, device=torch.device('cuda'))
    demo        = nm.Demo(environment.width, environment.height, filename=demo_path)
    actions     = demo.u;
    state       = demo.s;

    """ ..................................... """
    done = False
    demo_name = demo_path.split('/')[-1].split('.')[0]
    out_path = im_path + demo_name + '/'
    if(not os.path.exists(out_path)):
       os.mkdir(out_path)    
        
    environment.render(save_image=True)

    if(len(actions) > 0): 
        while(not done and environment.t < len(actions)):
            _, _, done = environment.step(actions[environment.t,0:2], save_image=True, save_path=out_path)

        #print("________________________")
        #print(" Level " + str(demo.env))
        #environment.score(True)
        #print("________________________")
        """ ..................................... """
        
    else: 
        print("Warning: empty demonstration")


#-------------------------------------------------------
# main()
args = sys.argv
print(len(args))
if len(args) == 4:
   playback(args[1], args[2], args[3])
else:
   print("ERROR: 3 command line arguments required")
   print("[Usage] python play.py <path to environment file> <path to demonstration> <image output path>")
