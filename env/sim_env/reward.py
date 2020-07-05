import numpy as np
import math
import sys

DEBUG = False

def calc_dist(p1, p2):
    return np.linalg.norm(p1 - p2)

def unit_v(v):
    """ Returns the unit vector of the vector.  """
    return v / np.linalg.norm(v, axis=-1, keepdims=True)

def angle_between(v1, v2):
    v1_u = unit_v(v1)
    v2_u = unit_v(v2)
    dot = np.tensordot(v1_u, v2_u, axes=(-1,-1))
    return np.arccos(dot)

pidiv2 = math.pi / 2
def norm_angle_between(v1, v2):
    ''' Angle normalized up to pi/2 '''
    a = angle_between(v1, v2)
    if a > pidiv2:
        a = math.pi - a
    return a

# Reward implementation for task reach version 0
class Reward_reach_v0(object):
    def __init__(self, env):
        self.env = env
        self.dist_epsilon = 0.03

    def reset(self):
        ''' Called from the env reset function '''
        s = self.env.state
        dist = calc_dist(s.needle_tip_pos, s.cur_target_pos)
        self.last_dist = dist
        self.reset_dist = dist
        reward_txt = None

    def get_reward_data(self):
        reward = 0.
        reward_txt = None
        done = False
        s = self.env.state
        ls = self.env.last_state
        success = 0

        # Distance to target component
        dist = calc_dist(s.needle_tip_pos, s.cur_target_pos)

        # Make sure we never get further than reset distance
        if dist > 1.2 * self.reset_dist:
            done = True
            reward -= 5.

        # Check for end condition
        if dist <= self.dist_epsilon:
            reward_txt = "Success!"
            done = True
            reward += 5.
            success = 1

        # End if we've touched the skin
        # y is height going up
        z_diff = s.cur_target_pos[2] - s.needle_tip_pos[2]
        y_target = s.cur_target_pos[1]
        z_target = s.cur_target_pos[2]
        y_tissue = y_target + 0.75 * z_diff
        y_needle = s.needle_tip_pos[1]
        z_needle = s.needle_tip_pos[2]
        if y_needle <= y_tissue:
            #print "low needle: needle y:{} z:{} < tissue y: {}. target y:{}, z:{} ".format(
            #        y_needle, z_needle, y_tissue, y_target, z_target)
            if not done:
                reward -= 5.
                reward_txt = "Bump into tissue!"
                done = True

        d = self.last_dist - dist
        d *= 10

        reward += d

        # Reward for getting to target
        if self.env.last_state is not None and \
           self.env.state.cur_target > self.env.last_state.cur_target:
            reward += 1.

        self.last_dist = dist

        # Reduce for timestep
        reward -= 0.02

        # Check for collisions
        if ls is not None and \
            (ls.instr_collisions < s.instr_collisions or \
            ls.instr_endo_collisions < s.instr_endo_collisions):
              reward -= 1.
              done = True

        # Check for out of view
        if s.tools_out_of_view > 0:
            reward -= 5.
            reward_txt = "Tools out of view!"
            done = True

        if self.env.t >= self.env.max_steps:
            # Need penalty so don't choose to do nothing
            #reward -= 4.
            reward_txt = "Out of time!"
            done = True

        if not s.needle_grasped:
            reward_txt = "[{:02d}] XXX Needle dropped!".format(self.env.server_num)
            print reward_txt
            done = True
            reward -= 5.

        # Check for errors
        if s.error and self.env.get_save_mode() != 'play':
            self.env.error_ctr += 1
            if self.env.error_ctr >= self.env.max_error_ctr:
                reward_txt = "Error!"
                done = True

        return reward, done, reward_txt, success

class Reward_suture_simple(object):
    ''' A sparse reward '''
    def __init__(self, env):
        self.env = env
        self.success_count = 3

    def _needle_to_target_d(self, src=0, dst=0):
        ss = self.env.state
        needle = ss.needle_points_pos
        dist = calc_dist(needle[src], self.targets[dst])
        return dist

    def reset(self):
        ss = self.env.state
        self.targets = np.array([ss.cur_target_pos, ss.next_target_pos])
        dist = self._needle_to_target_d()
        self.reset_dist = dist
        self.last_dist = dist
        self.success_count = 3

    def _check_basic_errors(self):
        ss = self.env.state
        ls = self.env.last_state
        nstatus = ss.needle_insert_status
        tstatus = ss.target_insert_status
        done = False
        reward = 0.
        reward_txt = ''

        if tstatus == -1 or \
           ss.outside_insert_radius or \
           ss.outside_exit_radius:
            done = True
            reward_txt = "Outside insert/exit radius!"
            reward -= 2.

        if ls is not None and \
            (ss.excessive_needle_pierces > ls.excessive_needle_pierces or \
            ss.excessive_insert_needle_pierces > ls.excessive_insert_needle_pierces or \
            ss.excessive_exit_needle_pierces > ls.excessive_exit_needle_pierces or \
            ss.incorrect_needle_throws > ls.incorrect_needle_throws):
            done = True
            reward_txt = "Excessive_needle_throw/pierce!"
            reward -= 2.

        # Check for collisions
        '''
        if ls is not None and \
            (ls.instr_collisions < ss.instr_collisions or \
            ls.instr_endo_collisions < ss.instr_endo_collisions):
              reward -= 2.
              reward_txt = "Instr collision!"
              done = True
        '''

        # check for errors
        if ss.error and self.env.get_save_mode() != 'play':
            self.env.error_ctr += 1
            if self.env.error_ctr >= self.env.max_error_ctr:
                reward_txt = "Error!"
                done = True

        # Check for out of view
        if ss.tools_out_of_view > 0:
            reward_txt = "Tools out of view!"
            reward -= 2.
            done = True

        if not ss.needle_grasped:
            print "[{:02d}] XXX Needle dropped!".format(self.env.server_num)
            reward_txt = "Needle dropped!"
            done = True
            reward -= 2.

        if not done and nstatus != tstatus:
            # Insert in wrong place
            txt = "[{:02d}] Mismatch ns:{}, ts:{}, lns:{}, lts:{}".format(
                    self.env.server_num, nstatus, tstatus,
                    ls.needle_insert_status if ls else -1, ls.target_insert_status if ls else -1)
            print txt
            reward_txt = txt
            done = True
            reward -= 2.

        if self.env.t >= self.env.max_steps:
            reward_txt = "Out of time"
            done = True

        return reward, done, reward_txt

    def _get_dist(self):
        ss = self.env.state
        tstatus = ss.target_insert_status
        dist = 0.
        if tstatus == 0:
            dist = self._needle_to_target_d()
        elif tstatus == 1:
            # I think it makes more sense to minimize dist to avg of both distances
            dist1 = self._needle_to_target_d(src=-1, dst=0) # dist to entry
            dist2 = self._needle_to_target_d(src=0, dst=1) # dist to target
            # Weight dist to target more heavily
            dist = 0.2 * dist1 + 0.8 * dist2
        elif tstatus == 2:
            dist = -self._needle_to_target_d(src=0, dst=1)
        elif tstatus == 3:
            dist = -self._needle_to_target_d(src=0, dst=1)
        return dist


    def get_reward_data(self):
        ss = self.env.state
        ls = self.env.last_state
        nstatus = ss.needle_insert_status
        tstatus = ss.target_insert_status
        needle = ss.needle_points_pos
        reward_txt = ''

        success = tstatus

        reward, done, reward_txt = self._check_basic_errors()

        if not done:
            # Get into tstatus 1 quickly
            if self.env.t >= 20 and tstatus == 0:
                reward_txt = "TStatus 0, too long!"
                done = True
                reward -= 5.

            dist = self._get_dist()
            if (tstatus == 0 and dist > self.reset_dist * 1.2) or \
                dist > self.reset_dist * 8.:
                reward_txt = "Too far!"
                done = True
                reward -= 5.

            if ls is not None:
                last_tstatus = ls.target_insert_status
                if tstatus > last_tstatus:
                    # progress, but don't reward for dist change
                    reward += 5.
                    reward_txt = "TStatus increase"

                    if tstatus >= 2:
                        if self.success_count <= 0:
                            done = True
                            reward += 5.
                            reward_txt = "Success!"
                        else:
                            self.success_count -= 1

                    #print "ts>ls, r={}".format(reward) #debug
                elif tstatus == last_tstatus:
                    # Check for change of dist
                    d = self.last_dist - dist
                    # Make negative moves worse
                    if d < 0:
                        d *= 2
                    if tstatus == 0:
                        reward += d
                    else:
                        reward += d * 100 # was 10
                    reward_txt += "delta: {:.5f}".format(d)
                    #print "ts=ls, r={}, d={}".format(reward, dist) #debug
                else:
                    # regression. no good
                    reward_txt = "Regression!"
                    reward -= 8.
                    done = True
                    #print "ts=ls, r={}".format(reward) #debug

                self.last_dist = dist

            # NOTE: was 0.05, tried doubling
            reward -= 0.01

        return reward, done, reward_txt, success


class Reward_suture_v1(Reward_suture_simple):
    def __init__(self, *args, **kwargs):
        super(Reward_suture_v1, self).__init__(*args, **kwargs)

    def reset(self):
        ss = self.env.state
        ls = self.env.last_state

        needle_pts = ss.needle_points_pos

        self.needle_r = ss.curvature_radius
        # needle segment lengths
        needle_lengths = []
        last_p = needle_pts[0]
        for p in needle_pts[1:]:
            needle_lengths.append(calc_dist(p, last_p))
            last_p = p
        self.needle_lengths = np.array(needle_lengths)

        # Ideally, but make sure targets are reset
        self.targets = np.array([ss.cur_target_pos, ss.next_target_pos])

        dist = calc_dist(ss.needle_tip_pos, self.targets[0])
        self.last_dist = dist
        self.reset_dist = dist

        # Get surface normal
        v1 = ss.tissue_corners[0] - ss.tissue_corners[1]
        v2 = ss.tissue_corners[0] - ss.tissue_corners[2]
        self.surf_norm = unit_v(np.cross(v1, v2))
        self.surf_pt = ss.tissue_corners[0]
        #self.surf_norm = np.array([0,1,0]) # approximate norm

        # Compute ideal circle center
        target_diam = calc_dist(self.targets[0], self.targets[1])
        height = calc_dist(self.needle_r, target_diam/2)
        self.circle_pt = \
            (self.targets[0] + self.targets[1])/2 + \
            self.surf_norm * height

        # Compute ideal circle plane vector
        v1 = self.targets[0] - self.targets[1]
        #v2 = np.array([0, -1, 0], dtype=np.float32)
        v2 = -self.surf_norm
        self.circle_v = unit_v(np.cross(v1, v2))

        self.last_dist_ideal = None
        self.last_a_ideal = None

    def _get_submerged_points(self, points):
        vectors = points - self.surf_pt
        angles = angle_between(self.surf_norm, vectors)
        submerged = angles > pidiv2
        return submerged

    def _submerged_s(self, submerged):
        l = ['T' if x else 'F' for x in submerged]
        return '[' + ','.join(l) + ']'

    def _get_needle_dist(self):
        ''' Find the point of the needle we care about most.
            If we're submerging, it's the point above the surface.
            If we're exiting, it's the point under the surface.
        '''
        ok = True
        ss = self.env.state
        tstatus = ss.target_insert_status
        needle = ss.needle_points_pos

        if tstatus == 0:
            dist = calc_dist(needle[0], self.targets[0])
        else:
            #from rl.utils import ForkablePdb
            #ForkablePdb().set_trace()

            submerged = self._get_submerged_points(needle)

            if tstatus == 1: # only entry
                # Check which points have y lower than target
                #submerged = needle[:,1] <= self.targets[0,1]
                idxs = np.where(submerged == False)[0]
                if len(idxs) == 0: # Fully submerged
                    #last_sub = len(needle) - 1
                    # Fully submerged? Impossible in tstatus 1
                    ok = False
                    text = "[{}] Error: tstatus 1 but all submerged, {}".format(
                            self.env.server_num, self._submerged_s(submerged))
                    print text
                    self.env.render(text=text)
                    return 0., False

                first_unsub = idxs[0]
                #if first_unsub == 0:
                #    raise ValueError("[{}] Error: tstatus 1 but no submerged points".
                #            format(self.env.server_num))

                if np.any(submerged[first_unsub:]):

                    #from rl.utils import ForkablePdb
                    #ForkablePdb().set_trace()
                    ok = False

                    text = "[{}] Error: tstatus 1: found submerged " \
                        "in wrong place: {}".format(self.env.server_num, self._submerged_s(submerged))
                    print text
                    self.env.render(text=text)

                # Relevant dist is to entry point
                dist = calc_dist(needle[first_unsub], self.targets[0])
                # Add the length of the segments not submerged
                extra_dist = np.sum(self.needle_lengths[first_unsub:])
                dist += extra_dist

            elif tstatus in [2, 3]: # entry and exit/exit
                # Check which points have y lower than exit target
                #submerged = needle[:,1] <= self.targets[1,1]
                idxs = np.where(submerged)[0]

                if len(idxs) == 0:
                    ok = False
                    text = "[{}] Error: tstatus {} but no submerged points: ".format(
                            self.env.server_num, tstatus, self._submerged_s(submerged))
                    print text
                    self.env.render(text=text)

                first_sub = idxs[0]
                dist = calc_dist(needle[first_sub], self.targets[1])
                extra_dist = np.sum(self.needle_lengths[first_sub:])
                dist += extra_dist
            else:
                ok = False
                print "Error: tstatus 4 not yet supported"

        return dist, ok

    def get_reward_data(self):

        ss = self.env.state
        ls = self.env.last_state
        nstatus = ss.needle_insert_status
        tstatus = ss.target_insert_status

        done = False
        reward = 0.
        d_a_ideal, d_dist_ideal = 0., 0.

        # ----- Components that measure alignment with target circle -----
        # Compute needle plane vector
        v1 = ss.needle_points_pos[0] - ss.needle_points_pos[1]
        v2 = ss.needle_points_pos[0] - ss.needle_points_pos[2]
        needle_v = unit_v(np.cross(v1, v2))

        # Compute avg dist from circle pt
        to_circle_pt = ss.needle_points_pos - self.circle_pt
        mid_dist = np.linalg.norm(to_circle_pt, axis=-1).mean()

        # Compute deviation from vector of ideal circle plane
        a_ideal = norm_angle_between(needle_v, self.circle_v)
        # Compute difference from radius (0.06)
        dist_ideal = mid_dist - self.needle_r

        if self.last_a_ideal is not None:
            d_a_ideal = self.last_a_ideal - a_ideal
            d_dist_ideal = self.last_dist_ideal - dist_ideal
        self.last_a_ideal = a_ideal
        self.last_dist_ideal = dist_ideal
        # -------------------------------------------

        done2, reward2 = self._check_basic_errors()
        done = done or done2
        reward += reward2

        if done:
            return reward, done

        dist, is_ok = self._get_needle_dist()
        if not is_ok:
            reward -= 2.
            done = True

        # Make sure we never get further than reset distance
        # In tstatus 0
        if tstatus == 0 and dist > 20. * self.reset_dist:
            done = True
            reward -= 2.

        elif dist > 20. * self.reset_dist:
            done = True
            reward -= 2.

        # Compute distance of next point from surface
        d = 0.
        if ls is not None:
            last_tstatus = ls.target_insert_status
            if tstatus > last_tstatus:
                # progress, but don't reward for dist change
                reward += 5.
            elif tstatus == last_tstatus:
                # Check for change of dist
                d = self.last_dist - dist
            else:
                # regression. no good
                reward -= 5.
                done = True

            self.last_dist = dist

        #from rl.utils import ForkablePdb
        #ForkablePdb().set_trace()

        reward += (d + d_a_ideal + d_dist_ideal * 10)

        if not done and tstatus == 0:
            # Don't forgive regressions in status 0
            if d < 0:
                done = True
                reward -= 2.

        if tstatus == 3: # Goal for now
            reward += 5.
            done = True

        # Reduce for timestep
        reward -= 0.02

        return reward, done

