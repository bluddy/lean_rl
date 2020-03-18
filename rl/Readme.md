# Needle Master tools
This repo contains implementation of DDPG and TD3 on Needle Master. <br>
To run: 
- 'python -m  TD3.main_image  [envionment]  [agent]  [input]' <br> 
-  eg: 'python -m  TD3.main_image  data/environment_14.txt  ddpg --mode=rgb_array'

# Gate position modification functionality
This implementation includes functionality that modifies gate position by training time. This functionality is only valid for environment_1.
To run:
- 'python -m TD3.main_image_move data/environment_1.txt [agent]  [input]'


