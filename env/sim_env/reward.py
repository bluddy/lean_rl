import numpy as np

DEBUG = False

def calc_dist(p1, p2):
    return np.linalg.norm(p1 - p2)

def unit_v(v):
    """ Returns the unit vector of the vector.  """
    return v / np.linalg.norm(v)

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
        s = self.env.state
        ls = self.env.last_state

        self.needle_r = s.curvature_radius

        self.targets = [np.array(s.cur_target_pos), np.array(s.next_target_pos)]

        dist = calc_dist(s.needle_tip_pos, self.targets[0])
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

    def get_reward_and_done(self):

        from rl.utils import ForkablePdb
        ForkablePdb().set_trace()

        ss = self.env.state
        ls = self.env.last_state

        # Compute needle plane vector
        v1 = ss.needle_points_pos[0] - ss.needle_points_pos[1]
        v2 = ss.needle_points_pos[0] - ss.needle_points_pos[2]
        v_needle = np.cross(v1, v2)

        # Compute avg dist from circle pt
        vectors = ss.needle_points_pos - self.circle_pt
        dist = np.linalg.norm(vs, axis=-1).mean()

        # Compute deviation from vector of ideal circle
        # Compute difference from radius (0.06)


# Reward implementation for task suture version 1
# Note that follow dot by dot is not ideal, because in case it overshoot,
# the reward can be bigger (it then set target to be the next one)
class Reward_suture_v1(object):
    def __init__(self, env):
        self.env = env
        self.circle_dist_target = 0.5
        self.angle_target = 0.2

    def reset(self):        # It will be called from env.reset from the very beginning
        #from rl.utils import ForkablePdb
        #ForkablePdb().set_trace()

        s = self.env.state
        ls = self.env.last_state

        dist = calc_dist(s.needle_tip_pos, s.next_target_pos)
        self.last_dist = dist
        self.reset_dist = dist

        self.needle_radius = s.curvature_radius

        # Target info
        self.insert_t = s.cur_target_pos
        self.exit_t = s.next_target_pos
        self.surf_norm = np.array([0,1,0])

        if self.debug:
            print("Insert_t {}, exit_t {}"
            .format(self.insert_t, self.exit_t))

        # Compute ideal circle center
        target_dist = calc_dist(self.insert_t, self.exit_t)
        h = np.sqrt(self.needle_radius**2 - (target_dist/2)**2)
        self.circle_center = \
            (self.insert_t + self.exit_t)/2 + self.surf_norm * h

        self.last_circle_dist = self._calc_circle_dist()
        self.reset_circle_dist = self.last_circle_dist

        self.last_angle_btw_3rd_pt_and_exit = self._calc_angle_btw_3rd_pt_and_exit()
        self.reset_angle_btw_3rd_pt_and_exit = self.last_angle_btw_3rd_pt_and_exit

    def _calc_circle_dist(self):
        # Needle info
        s = self.env.state
        needle_points_pos = s.needle_points_pos
        needle_tip_wrld = self.env.state.needle_tip_pos

        # Note that we need to initialize the needle tip to be close
        # to the insert point
        overall_circle_dist = 0

        for point_in_needle in s.needle_points_pos:
            # Compute distance from each point to ideal circle center
            dist_to_ideal_center = np.linalg.norm(point_in_needle - self.circle_center)
            dist_to_ideal_cicle = np.absolute(dist_to_ideal_center - self.needle_radius)

            # Compute distance metic
            tip_center = point_in_needle - self.circle_center

            # Normal direction to the ideal circle plane
            dir_norm = np.cross(self.surf_norm, (self.insert_t-self.exit_t) / np.linalg.norm(self.insert_t - self.exit_t))
            deviation_out_of_plane = abs(np.dot(dir_norm, tip_center)) * 10 # to convert to centimeters

            # Projection on the ideal circle plane
            dir_proj = np.cross(np.cross(dir_norm, tip_center / np.linalg.norm(tip_center)), dir_norm)
            deviation_in_plane = abs(abs(np.dot(tip_center, dir_proj)) - self.needle_radius) * 10 # to convert to centimeters

            # Accumulate dist
            this_point_dist = np.sqrt(deviation_in_plane**2+deviation_out_of_plane**2)
            overall_circle_dist += this_point_dist

            #if self.debug: print("The normal direction is {}".format(dir_norm))
            #if self.debug: print("This point has deviation in plane of {:.2f}".format(deviation_in_plane))
            #if self.debug: print("This point has deviation out of plane of {:.2f}".format(deviation_out_of_plane))

        return overall_circle_dist

    def _calc_angle_btw_3rd_pt_and_exit(self):

        def angle_between(v1, v2):
            """ Returns the angle in radians between vectors 'v1' and 'v2'::

                    >>> angle_between((1, 0, 0), (0, 1, 0))
                    1.5707963267948966
                    >>> angle_between((1, 0, 0), (1, 0, 0))
                    0.0
                    >>> angle_between((1, 0, 0), (-1, 0, 0))
                    3.141592653589793
            """

            v1_u = unit_vector(v1)
            v2_u = unit_vector(v2)
            return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

        s = self.env.state
        # Compute angle between center to 3rd needle point and center to exit
        from_center_to_3rd_needle_point = s.needle_points_pos[2] - self.circle_center
        from_center_to_exit = self.exit_t - self.circle_center
        angle = angle_between(from_center_to_3rd_needle_point, from_center_to_exit)
        return angle

    def get_reward_and_done(self):
        s = self.env.state
        ls = self.env.last_state

        # Distance to target component
        dist = calc_dist(s.needle_tip_pos, s.cur_target_pos)

        if self.debug:
            print("Dist from {} to {} is {}"
                .format(s.needle_tip_pos, self.insert_t, 10*dist))

        # Make sure we never get further than reset distance
        if dist > self.reset_dist * 50:
            if self.debug:
                print("d {:.3f} > reset_d {:.3f}!".format(dist, self.reset_dist))
            return -1., True

        # Check for collisions
        if ls.instr_collisions < s.instr_collisions or \
            ls.instr_endo_collisions < s.instr_endo_collisions:
            if self.debug:
                print("Collision! Instrument: {:.2f}, Endoscope: {:.2f}" \
                .format(s.instr_collisions, s.instr_endo_collisions))
            return -1., True

        # Check for out of view
        if s.tools_out_of_view > 0:
            if self.debug:
                print("tool is out of view")
            return -1, True

        if not s.needle_grasped:
            print "[{:02d}] XXX Needle dropped!".format(self.env.server_num)
            return -1, True

        reward = 0.0
        done = False

        circle_dist = self._calc_circle_dist()
        angle_btw_3rd_pt_and_exit = self._calc_angle_btw_3rd_pt_and_exit()

        d_circle_dist = self.last_circle_dist - circle_dist
        d_angle_btw_3rd_pt_and_exit = self.last_angle_btw_3rd_pt_and_exit - angle_btw_3rd_pt_and_exit

        self.last_circle_dist = circle_dist
        self.last_angle_btw_3rd_pt_and_exit = angle_btw_3rd_pt_and_exit

        #reward += d_circle_dist # for now, don't include the distance
        reward += d_angle_btw_3rd_pt_and_exit

        if self.debug:
            print("Dist to ideal circle is {:.2f}, decreased by {:.2f} (cm) -> reward of {:.2f}" \
                .format(circle_dist, d_circle_dist, d_circle_dist))
            print("Dist to ideal angle is {:.2f}, decreased by {:.2f} (radius) -> reward of {:.2f}" \
                .format(angle_btw_3rd_pt_and_exit, d_angle_btw_3rd_pt_and_exit, d_angle_btw_3rd_pt_and_exit))


        if self.env.last_state and self.env.state.needle_insert_status > self.env.last_state.needle_insert_status:
            print("On itertation {}: INSERT/Exit! current status: {:.2f}, previous status: {:.2f} -> reward of 1" \
                .format(self.env.t, self.env.state.needle_insert_status, self.env.last_state.needle_insert_status))
            reward += 1.

        elif self.env.last_state and self.env.state.needle_insert_status > self.env.last_state.needle_insert_status:
            print("Insert status going backword!! current status: {:.2f}, previous status: {:.2f} -> reward of -1" \
                .format(self.env.state.needle_insert_status, self.env.last_state.needle_insert_status))
            return -1, True

        # It's done when the needle goes out and align to plane and align to angle
        if (self.env.state.needle_insert_status==2) \
            and angle_btw_3rd_pt_and_exit<self.angle_target:
            #and circle_dist < self.circle_dist_target \ # for now, don't include the distance

            print("Successfully complete the mission!!")
            reward += 1.
            done = True

        if self.env.t >= self.env.max_steps:
            print("Reach max step")
            done = True
        # Check for errors
        if self.env.state.error:
            print("State error happens!")
            self.env.error_ctr += 1
            if self.env.error_ctr >= self.env.max_error_ctr:
                if self.debug: print("Error total reaches max of {:.2f}".format(self.env.max_error_ctr))
                done = True

        return reward, done
