import os, glob
from os.path import join as pjoin
import joblib
import subprocess as sp
import numpy as np
import random
import copy

class SaveMode(object):
    pass

class CommonEnv(object):
    def __init__(self, save_mode_path=None,
            server_num=0,
            save_mode='record',
            save_mode_play_ratio=0):

        self.server_num = server_num
        self._sm = SaveMode()
        self._sm.mode = None # record/play/empty
        self._sm.path = save_mode_path
        self._sm.ep = 0 # Episode for sim save
        self._sm.images = [] # Before dumping to file
        self._sm.states = []
        self._sm.play_ratio = save_mode_play_ratio
        self.set_save_mode(save_mode)
        self._sm.play_resets = 0
        self._sm.resets = 0


    def _gen_save_name(self, sim_save=False, use_episode=True, use_t=True):
        infix = ''
        if use_episode:
            infix = '_{:07d}'.format(self._sm.ep if sim_save else self.episode)
        suffix = ''
        if use_t:
            suffix = '_{:03d}'.format(self.t)
        return '{:02d}{}{}'.format(self.server_num, infix, suffix)

    def _step_play(self):
        ''' Take a fake step in the playback
            Assume save_mode = record
        '''

        if self.t + 1 > len(self._sm.states):
            from rl.utils import ForkablePdb
            ForkablePdb().set_trace()

        self.state = self._sm.states[self.t]
        self.image = self._sm.images[self.t]

        # We can't play beyond the last step
        play_done = self.t + 1 >= len(self._sm.states)

        return self.state.action, play_done

    def _step_record(self):
        ''' Save the image for future replays '''
        if self._sm.mode == 'record':
            self.render(save_path=self._sm.path, sim_save=True)
            self._sm.states.append(copy.deepcopy(self.state))

    def _load_sim_ep_images(self, img_width, img_height):
        save_name = self._gen_save_name(sim_save=True, use_t=False) + '.mp4'
        save_path = pjoin(self._sm.path, save_name)

        # Open video file
        command = [ 'ffmpeg',
            '-loglevel', '8',
            '-i', save_path,
            '-f', 'image2pipe',
            '-pix_fmt', 'rgb24',
            '-vcodec', 'rawvideo', '-'
            ]
        pipe = sp.Popen(command, stdout=sp.PIPE, bufsize=10**8)
        images = []
        while True:
            raw_image = pipe.stdout.read(img_width * img_height * 3)
            if len(raw_image) == 0:
                break
            image = np.fromstring(raw_image, dtype='uint8')
            image = image.reshape((img_height, img_width, 3))
            images.append(image)

        self._sm.images = images

    def _load_sim_ep_states(self):
        ''' Load episode states for replay '''

        while True:
            self._sm.ep = random.randint(1, self._sm.max_ep)

            save_name = self._gen_save_name(sim_save=True, use_t=False) + '.data'
            save_path = pjoin(self._sm.path, save_name)

            #print "Searching for ", save_path # debug

            if os.path.exists(save_path):

                self._sm.states = []
                try:
                    self._sm.states = joblib.load(save_path)
                except:
                    continue

                # A proper episode must have at least 2 steps: reset and a step
                if len(self._sm.states) > 1:
                    return True

            #print save_path, "not found"

    def _dump_sim_ep_states(self):
        ''' Save episode state from memory for future replay '''
        save_name = self._gen_save_name(sim_save=True, use_t=False) + '.data'
        joblib.dump(self._sm.states, pjoin(self._sm.path, save_name))
        self._sm.states = []

    def _update_save_info_file(self):
        ''' Update info file with latest episode '''
        info_file = pjoin(self._sm.path, 'info{}.txt'.format(self.server_num))
        with open(info_file, 'w') as file:
            file.write(str(self._sm.ep))

    def _read_save_info_file(self):
        # Find latest episode for this server num and update
        info_file = pjoin(self._sm.path,
            'info{}.txt'.format(self.server_num))
        if os.path.exists(info_file):
            with open(info_file) as f:
                content = f.read()
                self._sm.max_ep = int(content)
        else:
            self._sm.max_ep = 0

    def set_save_mode(self, save_mode):
        # Set whether we save sim data
        if self._sm.path is None:
            return

        last_save_mode = self._sm.mode
        self._sm.mode = save_mode

        #print "[{:02d}] save mode: {}, cur: {}, last: {} ".format(self.server_num,
        #    v, self._sm.mode, last_save_mode) # debug


        if self._sm.mode == 'record':
            if not os.path.exists(self._sm.path):
                os.makedirs(self._sm.path)

            self._sm.states = []

            # Eliminate all leftover png files for this server
            save_name = self._gen_save_name(sim_save=True,
                use_t=False, use_episode=False)
            pattern = os.path.join(self._sm.path, save_name + '*.png')
            for f in glob.glob(pattern):
                os.remove(f)

            self._read_save_info_file()
            self._sm.ep = self._sm.max_ep + 1

        elif self._sm.mode == 'play':
            self._read_save_info_file()

    def _reset_play(self, img_width, img_height):
        ''' Load up an episode for playback
            Assumption: save_mode = 'play'
        '''
        good = False
        while not good:
            self._load_sim_ep_states()

            self._load_sim_ep_images(img_width=img_width, img_height=img_height)

            if len(self._sm.images) == len(self._sm.states):
                good = True
            else:
                print "[{}] Mismatch! save mode images:{} states:{} epsiode:{}".format(
                    self.server_num, len(self._sm.images),
                    len(self._sm.states), self._sm.ep)

        self.state = self._sm.states[0]
        self.image = self._sm.images[0]
        return True

    def _reset_try_play(self, img_width, img_height):
        if self._sm.play_ratio > 0:
            if self._sm.mode == 'play':
                self._sm.resets += 1
                if self._sm.resets >= self._sm.play_ratio:
                    self._sm.resets = 0
                    self.set_save_mode('real')
            elif self._sm.mode == 'real':
                self._sm.resets += 1
                if self._sm.resets >= 1:
                    self._sm.resets = 0
                    self.set_save_mode('play')

        if self._sm.mode == 'play':
            self._reset_play(img_width, img_height)
            return True
        else:
            return False

    def _reset_record(self):
        if self._sm.mode == 'record':
            # Handle previous episode
            if len(self._sm.states) > 0:
                self._dump_sim_ep_states() # dump states from memory
                # Convert last episode's images to video
                self.convert_to_video(save_path=self._sm.path, sim_save=True)
                self._update_save_info_file() # update latest episode to last one
                if self._sm.ep > self._sm.max_ep:
                    self._sm.max_ep = self._sm.ep

                # Handle new episode
                self._sm.ep += 1

            # New episode
            self._step_record()

    def convert_to_video(self, save_path='./out', sim_save=False):
        ''' Convert to video and delete image files '''
        save_name = self._gen_save_name(sim_save=sim_save, use_t=False)
        pattern = pjoin(save_path, save_name + '*.png')
        out_file = pjoin(save_path, save_name + '.mp4')
        filelist = glob.glob(pattern)
        if len(filelist) > 0:
            cmd = 'cat {} | ffmpeg -loglevel 8 -f image2pipe -r 5 -vcodec png -i - -vcodec libx264 -preset veryslow {}'.format(pattern, out_file)
            os.system(cmd)
            for f in filelist:
                os.remove(f)
        else:
            print "convert_to_video: No files found for ", pattern

    def get_save_mode(self):
        return self._sm.mode


