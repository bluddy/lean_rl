import numpy as np
import math
import sys

DEBUG = False

def calc_dist(p1, p2):
    return np.linalg.norm(p1 - p2)

def unit_v(v):
    """ Returns the unit vector of the vector.  """
    return v / np.linalg.norm(v)

def angle_between(v1, v2):
    v1_u = unit_v(v1)
    v2_u = unit_v(v2)
    return np.arccos(np.dot(v1_u, v2_u))

pi2 = math.pi / 2
def norm_angle_between(v1, v2):
    ''' Angle normalized up to pi/2 '''
    a = angle_between(v1, v2)
    if a > pi2:
        a = math.pi - a
    return a

# Reward implementation for task reach version 0
class Reward_reach_v0(object):
    def __init__(self, env):
        self.env = env
        self.dist_epsilon = 0.025

    def reset(self):
        ''' Called from the env reset function '''
        s = self.env.state
        dist = calc_dist(s.needle_tip_pos, s.cur_target_pos)
        self.last_dist = dist
        self.reset_dist = dist

    def get_reward_and_done(self):
        reward = 0.
        done = False
        s = self.env.state
        ls = self.env.last_state

        # Distance to target component
        dist = calc_dist(s.needle_tip_pos, s.cur_target_pos)

        # Make sure we never get further than reset distance
        if dist > 1.2 * self.reset_dist:
            done = True
            reward -= 5.

        # Check for end condition
        if dist <= self.dist_epsilon:
            done = True
            reward += 2.

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
                reward -= 2.
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
        reward -= 0.05

        # Check for collisions
        if ls is not None and \
            (ls.instr_collisions < s.instr_collisions or \
            ls.instr_endo_collisions < s.instr_endo_collisions):
              reward -= 1.
              done = True

        # Check for out of view
        if s.tools_out_of_view > 0:
            reward -= 1.
            done = True

        if self.env.t >= self.env.max_steps:
            # Need penalty so don't choose to do nothing
            #reward -= 4.
            done = True

        if not s.needle_grasped:
            print "[{:02d}] XXX Needle dropped!".format(self.env.server_num)
            done = True
            reward -= 2.

        # Check for errors
        if s.error:
            self.env.error_ctr += 1
            if self.env.error_ctr >= self.env.max_error_ctr:
                done = True

        return reward, done

class Reward_suture_v0(object):
    def __init__(self, env):
        self.env = env

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

        self.surf_norm = np.array([0,1,0]) # approximate norm

        # Compute ideal circle center
        target_diam = calc_dist(self.targets[0], self.targets[1])
        height = calc_dist(self.needle_r, target_diam/2)
        self.circle_pt = \
            (self.targets[0] + self.targets[1])/2 + \
            self.surf_norm * height

        # Compute ideal circle plane vector
        v1 = self.targets[0] - self.targets[1]
        v2 = np.array([0, -1, 0])
        self.circle_v = unit_v(np.cross(v1, v2))

        self.last_dist_ideal = None
        self.last_a_ideal = None

    def _get_needle_dist(self):
        ''' Find the point of the needle we care about most.
            If we're submerging, it's the point above the surface.
            If we're exiting, it's the point under the surface.
        '''
        ss = self.env.state
        status = ss.needle_insert_status
        needle = ss.needle_points_pos

        if status == 0:
            dist = calc_dist(needle[0], self.targets[0])

        elif status == 1: # only entry
            # Check which points have y lower than target
            submerged = needle[:,1] <= self.targets[0,1]
            idxs = np.where(submerged == False)[0]
            if len(idxs) == 0: # Fully submerged
                #last_sub = len(needle) - 1
                # Fully submerged? Impossible in status 1
                raise ValueError("[{}] Error: status 1 but all submerged".
                        format(self.env.server_num))

            first_unsub = idxs[0]
            #if first_unsub == 0:
            #    raise ValueError("[{}] Error: status 1 but no submerged points".
            #            format(self.env.server_num))

            if np.any(submerged[first_unsub:]):

                from rl.utils import ForkablePdb
                ForkablePdb().set_trace()

                raise ValueError("[{}] Error: status 1: found submerged "
                    "in wrong place!".format(self.env.server_num))

            # Relevant dist is to entry point
            dist = calc_dist(needle[first_unsub], self.targets[0])
            # Add the length of the segments not submerged
            extra_dist = np.sum(self.needle_lengths[first_unsub:])
            dist += extra_dist

        elif status in [2, 3]: # entry and exit/exit
            # Check which points have y lower than exit target
            submerged = needle[:,1] <= self.targets[1,1]
            idxs = np.where(submerged)[0]

            if len(idxs) == 0:
                raise ValueError(
                    "Error: status {} but no submerged points".format(status))

            first_sub = idxs[0]
            dist = calc_dist(needle[first_sub], self.targets[1])
            extra_dist = np.sum(self.needle_lengths[first_sub:])
            dist += extra_dist
        else:
            raise ValueError("Error: status 4 not yet supported")

        return dist

    def get_reward_and_done(self):

        ss = self.env.state
        ls = self.env.last_state
        status = ss.needle_insert_status

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

        dist = self._get_needle_dist()

        # Make sure we never get further than reset distance
        # In status 0
        if status == 0 and dist > 20. * self.reset_dist:
            done = True
            reward -= 5.

        # Compute distance of next point from surface

        d = 0.
        if ls is not None:
            last_status = ls.needle_insert_status
            if status > last_status:
                # progress, but don't reward for dist change
                reward += 1.
            elif status == last_status:
                # Check for change of dist
                d = self.last_dist - dist
            else:
                # regression. no good
                reward -= 2.
                done = True

            self.last_dist = dist

        #from rl.utils import ForkablePdb
        #ForkablePdb().set_trace()

        reward += 10 * (d + d_a_ideal + d_dist_ideal * 10)

        if status == 3: # Goal for now
            reward += 1.
            done = True

        # Reduce for timestep
        reward -= 0.05

        # Check for out of view
        if ss.tools_out_of_view > 0:
            reward -= 1.
            done = True

        if self.env.t >= self.env.max_steps:
            done = True

        if not ss.needle_grasped:
            print "[{:02d}] XXX Needle dropped!".format(self.env.server_num)
            done = True
            reward -= 2.

        # Check for errors
        if ss.error:
            self.env.error_ctr += 1
            if self.env.error_ctr >= self.env.max_error_ctr:
                done = True

        return reward, done


