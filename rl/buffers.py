import numpy as np
import os
from os.path import join as pjoin
import joblib
import imageio
import random, time

# Code based on:
# https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py


# Expects tuples of (state, next_state, action, reward, done)
class ReplayBuffer(object):
    def __init__(self, mode, capacity):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0
        self.mode = mode
        self.use_priorities = False

    def add(self, data, *args, **kwargs):

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

        data = self._process_samples(samples)

        # Append columns: idx
        data.extend([None])

        return data

    def _process_samples(self, data):
        # list of lists of state, next_state etc
        batch = list(zip(*data))

        if self.mode == 'mixed':
            # Swap it so tuple is on the outside, batch is on inside
            s1 = [np.concatenate(b) for b in zip(*batch[0])]
            s2 = [np.concatenate(b) for b in zip(*batch[1])]
        else:
            s1 = np.concatenate(batch[0])
            s2 = np.concatenate(batch[1])

        a = np.array(batch[2], copy=False)
        r = np.array(batch[3], copy=False).reshape(-1, 1)
        d = np.array(batch[4], copy=False).reshape(-1, 1)
        sample = [s1, s2, a, r, d]

        # Process extra values
        if len(batch) > 5:
            sample.append(np.array(batch[5], copy=False))
        else:
            sample.append(None)

        return sample

    def update_priorities(self, x, y):
        pass

    def __len__(self):
        return len(self.buffer)

    def display(self):
        print("buffer len: ", len(self))

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
    def __init__(self, mode, capacity, prob_alpha=0.6,
            vacate=False, **kwargs):
        super(NaivePrioritizedBuffer, self).__init__(mode, capacity)
        self.prob_alpha = prob_alpha
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.vacate = vacate
        self.use_priorities = True

    def add(self, data, **kwargs):
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

        data = self._process_samples(samples)
        # Append columns
        data.append(indices)

        return data

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in list(zip(batch_indices, batch_priorities)):
            self.priorities[idx] = prio

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
    def __init__(self, mode, capacity, procs=1,
            calc_length=10000, gamma=0.99, clip=100.,
            sub_buffer='replay', **kwargs):
        '''
            @procs: number of processes to keep track of
        '''
        super(TieredBuffer, self).__init__(mode, capacity)

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

        self.buffers = [create_buf(mode=mode, capacity=capacity//3)
                for _ in range(num_tiers)]

        self.probs = np.array([0.32, 0.32, 0.36])

        # Probs of using the direct Q values
        self.q_probs = np.array([0., 0., 0.1])

        self.last_tier = None

    def add(self, data, num=0, *args, **kwargs):
        proc=num

        # data: [s1, s2, a, r, d]

        # Add to respective proc buffer
        self.proc_buffers[proc].append(data)
        r = data[3]
        self.proc_rewards[proc] += r
        #print("proc_buf len: ", len(self.proc_buffers[proc])) # debug

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
        print("max_R: {:.2f} up_R: {:.2f}, low_R: {:.2f}, clipped {}".format(
                self.stats.max(), self.stats.upper(), self.stats.median(),
                self.clip_count),end='')
        for i, buf in enumerate(self.buffers):
            print("{}:{}, ".format(i, len(buf)),end='')
        print("")

class MultiBuffer(ReplayBuffer):
    ''' Buffer that contains other buffers
        Buffer choice is using
    '''
    def __init__(self, mode, capacity, count=2, sub_buffer='priority', **kwargs):
        super(MultiBuffer, self).__init__(mode, capacity)

        self.count = count

        if sub_buffer == 'replay':
            create_buf = ReplayBuffer
        elif sub_buffer == 'priority':
            create_buf = NaivePrioritizedBuffer

        self.buffers = [create_buf(mode=mode, capacity=capacity//count)
                for _ in range(self.count)]

    def add(self, data, num=None, **kwargs):
        target = num % self.count

        #print("Adding to ", target) # debug
        self.buffers[target].add(data, **kwargs)

    def sample(self, batch_size, num=None, **kwargs):
        target = num % self.count

        #print("Sampling from ", target) # debug
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
        a = np.concatenate(batch[1])
        es = np.concatenate(batch[2])

        return [s, a, es]

    def sample(self, batch_size):
        assert(len(self) > 0)

        pick_size = len(self) if len(self) < batch_size else batch_size

        indices = np.random.choice(len(self), pick_size)

        samples = [self.buffer[idx] for idx in indices]

        data = self._process_samples(samples)

        return data
