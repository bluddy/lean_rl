import numpy as np
import os
from os.path import join as pjoin
import joblib
import imageio
import random, time

# Code based on:
# https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py

def state_compress(mode, state):
    ''' Should be called for compression mode '''

    def compress_img(d):
        l = []
        for i in range(0, len(d), 3):
            dd = d[i:i+3].transpose((1,2,0))
            l.append(imageio.imwrite(imageio.RETURN_BYTES, dd, format='PNG'))
        return l

    if mode == 'image':
        d = state.squeeze(0)
        return compress_img(d)

    elif mode == 'mixed':
        d = state[0].squeeze(0)
        s = compress_img(d)
        return (s, state[1])

def state_decompress(mode, state):
    ''' Should be called for compression mode '''

    def decompress_img(l):
        limg = []
        for d in l:
            limg.append(imageio.imread(d, format='PNG').transpose((2,0,1)))
        img = np.concatenate(limg, axis=0)
        img = np.expand_dims(img, 0)
        return img

    if mode == 'image':
        return decompress_img(state)

    elif mode == 'mixed':
        s = decompress_img(state[0])
        return (s, state[1])

# Expects tuples of (state, next_state, action, reward, done)
class ReplayBuffer(object):
    def __init__(self, mode, capacity, compressed=False):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0
        self.mode = mode
        # Buffers to reuse memory
        self.compressed = compressed

    def decompress(self, data):
        ''' Used when sampling
            @data: [state1, state2, ...]
        '''
        img1 = state_decompress(self.mode, data[0])
        img2 = state_decompress(self.mode, data[1])
        return [img1, img2] + data[2:]

    def add(self, data, *args, **kwargs):
        ''' In compressed mode, the states must be pre-compressed '''

        if len(self) < self.capacity:
            self.buffer.append(data)
        else:
            self.buffer[self.pos] = data

        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, *args, **kwargs):
        assert(len(self) > 0)

        pick_size = len(self) if len(self) < batch_size else batch_size

        indices = np.random.choice(len(self), pick_size)

        samples = [self.buffer[idx] for idx in indices]

        if self.compressed:
            samples = [self.decompress(s) for s in samples]

        data = self._process_samples(samples)

        # Append columns: idx, w
        data.extend([None, None])

        return data, 0.

    def _process_samples(self, data):
        # list of lists of state, next_state etc
        batch = zip(*data)

        if self.mode == 'mixed':
            # Swap it so tuple is on the outside, batch is on inside
            s1 = [np.concatenate(b) for b in zip(*batch[0])]
            s2 = [np.concatenate(b) for b in zip(*batch[1])]
        else:
            try:
                s1 = np.concatenate(batch[0])
                s2 = np.concatenate(batch[1])
            except:
                import pdb
                pdb.set_trace()

        a = np.array(batch[2], copy=False)
        r = np.array(batch[3], copy=False).reshape(-1, 1)
        d = np.array(batch[4], copy=False).reshape(-1, 1)
        sample = [s1, s2, a, r, d]

        # Process extra value
        if len(batch) > 5:
            x = np.array(batch[5], copy=False)
            sample.append(x)
        else:
            sample.append(None)


        return sample

    def update_priorities(self, x, y):
        pass

    def __len__(self):
        return len(self.buffer)

    def display(self):
        print "buffer len: ", len(self)

    def save_to_disk(self, file):
        if self.mode == 'image':
            d = np.transpose(data[0].squeeze(0), (1, 2, 0))
            imageio.imwrite(file + '1.png', d)
            d = np.transpose(data[1].squeeze(0), (1, 2, 0))
            imageio.imwrite(file + '2.png', d)
            joblib.dump(data[2:], file + '.data')
        elif self.mode == 'mixed':
            d = np.transpose(data[0][0].squeeze(0), (1, 2, 0))
            imageio.imwrite(file + '1.png', d)
            d = np.transpose(data[1][0].squeeze(0), (1, 2, 0))
            imageio.imwrite(file + '2.png', d)
            joblib.dump([data[0][1]] + [data[1][1]] + data[2:], file + '.data')
        elif self.mode == 'state':
            joblib.dump(data, file + '.data')

    def load_from_disk(self, file):
        if self.mode == 'image':
            s1 = imageio.imread(file + '1.png')
            s1 = np.expand_dims(np.transpose(s1, (2, 0, 1)), 0)
            s2 = imageio.imread(file + '2.png')
            s2 = np.expand_dims(np.transpose(s2, (2, 0, 1)), 0)
            x, y, z = joblib.load(file + '.data')
            data = [s1, s2, x, y, z]
        elif self.mode == 'mixed':
            s1 = imageio.imread(file + '1.png')
            s1 = np.expand_dims(np.transpose(s1, (2, 0, 1)), 0)
            s2 = imageio.imread(file + '2.png')
            s2 = np.expand_dims(np.transpose(s2, (2, 0, 1)), 0)
            s1_2, s2_2, x, y, z = joblib.load(file + '.data')
            data = [[s1, s1_2], [s2, s2_2], x, y, z]
        elif self.mode == 'state':
            s1, s2, x, y, z = joblib.load(file + '.data')
            data = [s1, s2, x, y, z]
        return data

class NaivePrioritizedBuffer(ReplayBuffer):
    def __init__(self, mode, capacity, compressed=False, prob_alpha=0.6,
            vacate=False, **kwargs):
        super(NaivePrioritizedBuffer, self).__init__(mode, capacity, compressed)
        self.prob_alpha = prob_alpha
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.vacate = vacate

    def add(self, data, **kwargs):
        ''' In compressed mode, the states must be pre-compressed '''

        max_prio = self.priorities.max() if self.buffer else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append(data)
            new_pos = self.pos
        else:
            if self.vacate:
                # vacate the lowest priorities
                min_idx = np.argmin(self.priorities)
                self.buffer[min_idx] = data
                new_pos = min_idx
            else:
                self.buffer[self.pos] = data
                new_pos = self.pos

        self.priorities[new_pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4, **kwargs):
        assert(len(self) > 0)

        pick_size = len(self) if len(self) < batch_size else batch_size

        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]

        probs = prios ** self.prob_alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self), pick_size, p=probs)

        # Get the weights
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        samples = [self.buffer[idx] for idx in indices]

        if self.compressed:
            samples = [self.decompress(s) for s in samples]

        data = self._process_samples(samples)
        # Append columns
        data.append(indices)
        data.append(weights)

        return data, 0.

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in list(zip(batch_indices, batch_priorities)):
            self.priorities[idx] = prio

class DiskReplayBuffer(ReplayBuffer):
    ''' Replay buffer that writes to disk, thus having unlimited storage space '''
    def __init__(self, mode, capacity, path, buffer_capacity=1000, **kwargs):
        super(DiskReplayBuffer, self).__init__(mode, capacity)
        self.path = pjoin(path, 'rbuffer')
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        self.length = 0 # Number of entries in disk buffer
        self.buffer = []
        self.buffer_capacity = buffer_capacity
        self.buf_offset = 0

    def add(self, data, *args, **kwargs):
        ''' In compressed mode, the states must be pre-compressed '''

        self.buffer.append(data)

        if len(self.buffer) > self.buffer_capacity:
            # Dump buffer to disk
            #print "dumping buffer. offset: ", self.buf_offset, " size: ", len(self.buffer) # Debug
            # TODO: Support mixed mode
            for i, data in enumerate(self.buffer):
                file = pjoin(self.path, 's_' + str(self.buf_offset + i))
                #joblib.dump(data, file)
                # debug test
                d = np.transpose(data[0].squeeze(0), (1, 2, 0))
                imageio.imwrite(file + '1.png', d)
                d = np.transpose(data[1].squeeze(0), (1, 2, 0))
                imageio.imwrite(file + '2.png', d)
                joblib.dump(data[2:], file + '.data')
            self.buf_offset = \
                    (self.buf_offset + len(self.buffer)) % self.capacity
            self.buffer = []

        if len(self) < self.capacity:
            self.length += 1

    def sample(self, batch_size, *args, **kwargs):
        ''' Sample from disk buffer '''
        assert(len(self) >= 0)

        pick_size = len(self) if len(self) < batch_size else batch_size

        indices = np.random.choice(len(self), pick_Size)

        #print "offset: ", self.buf_offset, "buflen: ", len(self.buffer), "len: ", len(self) # Debug

        samples = []
        for idx in indices:
            # Check if it's in our buffer
            if idx >= self.buf_offset and idx < self.buf_offset + len(self.buffer):
                samples.append(self.buffer[idx - self.buf_offset])
            else:
                file = pjoin(self.path, 's_' + str(idx))
                #data = joblib.load(file)
                # debug test
                s1 = imageio.imread(file + '1.png')
                s1 = np.expand_dims(np.transpose(s1, (2, 0, 1)), 0)
                s2 = imageio.imread(file + '2.png')
                s2 = np.expand_dims(np.transpose(s2, (2, 0, 1)), 0)
                x, y, z = joblib.load(file + '.data')
                data = (s1, s2, x, y, z)
                samples.append(data)

        data = self._process_samples(samples)
        # Append columns
        data.append([None, None])

        return data, 0.

    def __len__(self):
        return self.length

class StatCalc(object):
    ''' Rolling buffer with mean and max '''
    def __init__(self, size):
        self.buffer = np.zeros((size,), dtype=np.float32)
        self.sortbuf = np.zeros((size,), dtype=np.float32)
        self.pos = 0
        self.length = 0
        self._mean = 0.
        self.total = 0.
        self.max_idx = 0

    def add(self, x):

        # Adjust total
        self.total -= self.buffer[self.pos]
        self.total += x

        if self.length < len(self.buffer):
            self.length += 1
            self.sortbuf[self.length-1] = x
        else:
            old_val = self.buffer[self.pos]
            # Find old value's index
            idx = np.searchsorted(self.sortbuf, old_val)
            self.sortbuf[idx] = x

        self.buffer[self.pos] = x

        # Really timsort, handles pre-sorted data well
        self.sortbuf[:self.length].sort(kind='mergesort')

        self.pos = (self.pos + 1) % len(self.buffer)

    def median(self):
        return self.sortbuf[int(self.length * 0.7)]

    def upper(self):
        return self.sortbuf[int(self.length * 0.95)]

    def max(self):
        return self.sortbuf[self.length - 1]

class TieredBuffer(ReplayBuffer):
    def __init__(self, mode, capacity, compressed=False, procs=1,
            calc_length=10000, gamma=0.99, clip=100.,
            sub_buffer='replay', **kwargs):
        '''
            @procs: number of processes to keep track of
        '''
        super(TieredBuffer, self).__init__(mode, capacity, compressed)

        self.gamma = float(gamma) # For Q computation
        self.clip = clip
        self.clip_count = 0

        self.proc_buffers = [[] for _ in range(procs)]
        self.proc_rewards = [0. for _ in range(procs)]

        self.stats = StatCalc(calc_length)

        num_tiers = 3
        self.bufnums = [i for i in range(num_tiers)]

        if sub_buffer == 'replay':
            create_buf = ReplayBuffer
        elif sub_buffer == 'priority':
            create_buf = NaivePrioritizedBuffer

        self.buffers = [create_buf(mode=mode, capacity=capacity//3, compressed=compressed)
                for _ in range(num_tiers)]

        self.probs = np.array([0.32, 0.32, 0.36])

        # Probs of using the direct Q values
        self.q_probs = np.array([0., 0., 0.1])

        self.last_tier = None

    def add(self, data, num=0, *args, **kwargs):
        ''' In compressed mode, the states must be pre-compressed '''
        proc=num

        # data: [s1, s2, a, r, d]

        # Add to respective proc buffer
        self.proc_buffers[proc].append(data)
        r = data[3]
        self.proc_rewards[proc] += r
        #print "proc_buf len: ", len(self.proc_buffers[proc]) # debug

        # Check if we're done. If so, move to permanent buffer
        done = data[4]
        if done:
            reward = self.proc_rewards[proc]
            self.proc_rewards[proc] = 0.
            # Check if cap exceeded (bad rewards)
            if reward > self.clip:
                # Don't add to buffers
                self.clip_count += 1
            else:
                if reward >= self.stats.upper():
                    tier = 2
                elif reward >= self.stats.median():
                    tier = 1
                else:
                    tier = 0

                # Assign Q to every step
                q = float(0.)
                for [s1, s2, a, r, d] in reversed(self.proc_buffers[proc]):
                    q = r + self.gamma * q
                    self.buffers[tier].add([s1, s2, a, r, d, q])

                # Update statistics
                self.stats.add(reward)

            self.proc_buffers[proc] = [] # Clear proc buffer

    def sample(self, batch_size, *args, **kwargs):
        assert(len(self) > 0)

        # Roll to see which buffer we'll draw from
        selected = False
        while not selected:
            num = np.random.choice(self.bufnums, p=self.probs)
            if len(self.buffers[num]) > 0:
                selected = True

        self.last_tier = num
        sample, _ = self.buffers[num].sample(batch_size)
        q_prob = self.q_probs[num]
        return sample, q_prob

    def __len__(self):
        return sum((len(b) for b in self.buffers))

    def update_priorities(self, x, y):
        self.buffers[self.last_tier].update_priorities(x,y)

    def display(self):
        print "max_R: {:.2f} up_R: {:.2f}, low_R: {:.2f}, clipped {}".format(
                self.stats.max(), self.stats.upper(), self.stats.median(),
                self.clip_count),
        for i, buf in enumerate(self.buffers):
            print "{}:{}, ".format(i, len(buf)),
        print ""

class MultiBuffer(ReplayBuffer):
    ''' Buffer that contains other buffers
        Buffer choice is using
    '''
    def __init__(self, mode, capacity, compressed, count=2, sub_buffer='priority', **kwargs):
        super(MultiBuffer, self).__init__(mode, capacity, compressed)

        self.count = count

        if sub_buffer == 'replay':
            create_buf = ReplayBuffer
        elif sub_buffer == 'priority':
            create_buf = NaivePrioritizedBuffer

        self.buffers = [create_buf(mode=mode, capacity=capacity//count, compressed=compressed)
                for _ in range(self.count)]

    def add(self, data, num=None, **kwargs):
        ''' In compressed mode, the states must be pre-compressed '''
        target = num % self.count

        #print "Adding to ", target # debug
        self.buffers[target].add(data, **kwargs)

    def sample(self, batch_size, num=None, **kwargs):
        target = num % self.count

        #print "Sampling from ", target # debug
        assert(len(self.buffers[target]) > 0)

        return self.buffers[target].sample(batch_size, **kwargs)

    def __len__(self):
        return sum((len(b) for b in self.buffers))


class CNNBuffer(ReplayBuffer):
    ''' Buffer used only for CNN testing '''
    def __init__(self, *args, **kwargs):
        super(CNNBuffer, self).__init__(*args, **kwargs)

    def _process_samples(self, data):
        ''' Samples are very simple here: state and ideal action '''
        batch = zip(*data)

        s = np.concatenate(batch[0])
        a = np.array(batch[1])

        return [s, a]

    def sample(self, batch_size):
        assert(len(self) > 0)

        pick_size = len(self) if len(self) < batch_size else batch_size

        indices = np.random.choice(len(self), pick_size)

        samples = [self.buffer[idx] for idx in indices]

        if self.compressed:
            samples = [self.decompress(s) for s in samples]

        data = self._process_samples(samples)

        return data
