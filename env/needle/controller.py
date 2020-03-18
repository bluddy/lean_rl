'''

Classes for traditional controllers

written by Lifan 02/2019. Ported by Molly 03/2019


'''

class PIDcontroller:
    '''
            Proportional Differential Integral Controller

            Args:
                params: P parameters
                bounds: action constraints
    '''
    def __init__(self, params=None, bounds=None):
        self.filename = filename

        if(params is not None):
            self.params = np.array(params)
        else:
            self.params = np.array([1,1])

        self.bounds = bounds


    def step(self, cur_state, goal_state):
        cur_state  = self.convert_cur_state(cur_state)
        error = self.convert_goal_state(goal_state)
        action = np.multiply(error, self.params)

        # check we are within bounds of actions (if bounds were provided)
        # TODO: why do we only check dY?
        if(self.bounds is not None):
            if action[1] < -bounds[1]:
                action[1] = -bounds[1]
            elif action[1] > bounds[1]:
                action[1] = bounds[1]

        return action

    def convert_cur_state(self, cur_state):
        ''' flip the angle '''
        state = np.array([cur_state[0], cur_state[1], -1*cur_state[2]])
        return state

    def convert_goal_state(self, cur_state, gate_pos):
        '''
                Rotate goal state into coordinate frames of the needle using transform:

                R = [[cos(w) sin(w)],       t = [[-x],
                     [-sin(w) cos(w)]]           [-y]]

                where the current state is [x, y, w]

                Args:
                    cur_state: the needle state (after convert_cur_state has been called). length 3 vector [x,y,w]
                    gate_pos: length 2 vector [x_gate, y_gate], the center of the next goal location. If there is not another
                                gate then pass in None. (This will cue the alg to move right to finish the level)

        '''
        if(gate_pos is not None):
            gate_x = gate_pos.x; gate_y = gate_pos.y
            cur_x = cur_state[0]; cur_y = cur_state[1]; cur_w = cur_state[2]

            error_x = (gate_x - cur_x)*np.cos(cur_w) + (gate_y - cur_y)*np.sin(cur_w)
            error_y = -(gate_x - cur_x) *np.sin(cur_w) + (gate_y - cur_y)*np.cos(cur_w)

        else:
            error_x = 50
            error_y = 0

        return [-error_x,  -error_y] # return the negative to induce motion to close the error
