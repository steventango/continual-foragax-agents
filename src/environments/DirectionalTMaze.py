# Implement T-maze from https://papers.nips.cc/paper/2001/hash/a38b16173474ba8b1a95bcbc30d3b8a5-Abstract.html
# Note the following changes
#   wrong_goal_reward is -1 instead of -0.1
#   gamma is 0.95 instead of 0.98
#   observation is repeated along HW for (2, 2) unlike the original of (1, 1)
import numpy as np
import jax.numpy as jnp
from utils.rlglue import BaseEnvironment

class DirectionalTMaze():
    def __init__(self, corridor_length=10, seed=np.random.randint(int(1e5))):
        # Disambiguity: corridor_length is excluding the junction.
        self.rng = np.random.RandomState(seed)
        self.gamma = 0.95
        self.corridor_length = corridor_length
        self.x = 0
        self.y = 0
        self.direction = 'R'
        self.right_goal_reward = 4
        self.wrong_goal_reward = -1
        self.other_state_reward = -0.1
        
    def generate_goal(self):
        # True goal is up, False goal is down
        self.goal_is_up = self.rng.rand() < 0.5
        
    def get_sign_state(self):
        if (self.goal_is_up and self.direction == 'U') or ((not self.goal_is_up) and self.direction == 'D'):
            return np.array([[[1, 1, 0]]])
        elif self.direction == 'L':
            return np.array([[[0, 1, 0]]])
        else:
            return np.array([[[0, 0, 1]]])
        
    def get_corridor_state(self):
        if self.direction == 'U' or self.direction == 'D':
            return np.array([[[0, 1, 0]]])
        else:
            return np.array([[[0, 0, 1]]])
    
    def get_junction_state(self):
        if self.direction == 'R':
            return np.array([[[0, 1, 0]]])
        else:
            return np.array([[[0, 0, 1]]])
    
    def get_goal_state(self):
        return np.array([[[0, 1, 0]]])
    
    def is_at_sign(self):
        return self.x == 0
    
    def is_at_junction(self):
        return self.x == self.corridor_length and self.y == 0
    
    def is_at_goal(self):
        return self.x == self.corridor_length and self.y != 0
    
    def get_state(self):
        if self.is_at_junction():
            state = self.get_junction_state()
        elif self.is_at_goal():
            state = self.get_goal_state()
        elif self.is_at_sign():
            state = self.get_sign_state()
        else:
            state = self.get_corridor_state()
        
        return np.tile(state, (2,2,1))
        
    def get_reward(self):
        if self.is_successful():
            return self.right_goal_reward
        elif self.is_at_goal():
            return self.wrong_goal_reward
        else:
            return self.other_state_reward
        
    def is_successful(self):
        if self.is_at_goal():
            if (self.goal_is_up and self.y == 1) or ((not self.goal_is_up) and self.y == -1):
                return True
        return False
        
    def get_state_value(self, x = None):
        if x is None: x = self.x
        if self.is_at_goal():
            return 0
        ret = self.right_goal_reward
        for _ in range(self.corridor_length, x, -1):
            ret = self.other_state_reward + self.gamma * ret
        return ret
    
    def get_action_value(self, action, x = None):
        if x is None: x = self.x
        if self.is_at_goal():
            return 0
        
        match action:
            case 0:
                if x == self.corridor_length:
                    return self.right_goal_reward if self.goal_is_up else self.wrong_goal_reward
                else:
                    return self.other_state_reward + self.gamma * self.get_state_value(x)
                
            case 1:
                if x == self.corridor_length:
                    return self.other_state_reward + self.gamma * self.get_state_value(x)
                else:
                    return self.other_state_reward + self.gamma * self.get_state_value(x + 1)
                
            case 2:
                if x == self.corridor_length:
                    return self.right_goal_reward if not self.goal_is_up else self.wrong_goal_reward
                else:
                    return self.other_state_reward + self.gamma * self.get_state_value(x)
                
            case 3:
                if x == 0:
                    return self.other_state_reward + self.gamma * self.get_state_value(x)
                else:
                    return self.other_state_reward + self.gamma * self.get_state_value(x - 1)
                
            case _:
                return 0

    def get_action_values(self, x = None):
        return [self.get_action_value(a) for a in range(4)]
        
    def start(self):
        self.x = 0
        self.y = 0
        self.generate_goal()
        return jnp.float32(self.get_state())

    def step(self, action):
        # actions: (0) left turn, (1) forward, (2) right turn
        match action:
            case 0:
                match self.direction:
                    case 'R':
                        self.direction = 'U'
                    case 'U':
                        self.direction = 'L'
                    case 'L':
                        self.direction = 'D'
                    case 'D':
                        self.direction = 'R'
                return jnp.float32(self.get_state()), jnp.float32(self.get_reward()), False, False, self.get_info()

            case 1:
                match self.direction:
                    case 'U':
                        if self.is_at_junction():
                            self.y += 1
                            return jnp.float32(self.get_state()), jnp.float32(self.get_reward()), True, False, self.get_info()
                        else:
                            return jnp.float32(self.get_state()), jnp.float32(self.get_reward()), False, False, self.get_info()
                
                    case 'R':
                        if self.is_at_junction():
                            return jnp.float32(self.get_state()), jnp.float32(self.get_reward()), False, False, self.get_info()
                        else:
                            self.x += 1
                            return jnp.float32(self.get_state()), jnp.float32(self.get_reward()), False, False, self.get_info()
                            
                    case 'D':
                        if self.is_at_junction():
                            self.y -= 1
                            return jnp.float32(self.get_state()), jnp.float32(self.get_reward()), True, False, self.get_info()
                        else:
                            return jnp.float32(self.get_state()), jnp.float32(self.get_reward()), False, False, self.get_info()
                        
                    case 'L':
                        if self.is_at_sign():
                            return jnp.float32(self.get_state()), jnp.float32(self.get_reward()), False, False, self.get_info()
                        else:
                            self.x -= 1
                            return jnp.float32(self.get_state()), jnp.float32(self.get_reward()), False, False, self.get_info()
                    
            case 2:
                match self.direction:
                    case 'R':
                        self.direction = 'D'
                    case 'D':
                        self.direction = 'L'
                    case 'L':
                        self.direction = 'U'
                    case 'U':
                        self.direction = 'R'
                return jnp.float32(self.get_state()), jnp.float32(self.get_reward()), False, False, self.get_info()
                
            case _:
                raise NotImplementedError("Illegal action")
        
    def get_info(self):
        return {
            "gamma": self.gamma,
            "success": self.is_successful()
            }
