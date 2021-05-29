""" Required submission file

In this file, you should implement your `AgentSpec` instance, and **must** name it as `agent_spec`.
As an example, this file offers a standard implementation.
"""

import pickle
import os
path = os.path.split(os.path.realpath(__file__))[0]
import sys
sys.path.append(path)
import random

import gym

from pathlib import Path
import pickle
import gym

import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam, RMSprop, SGD
import os
from collections import deque
import numpy as np
from keras.layers.merge import concatenate
from keras.layers import Input, Dense, Conv2D, Flatten
from keras.models import Model

# contains all of the intersections


class GraphAgent():
    def __init__(self):

        # DQN parameters

        self.now_phase = {}
        self.green_sec = 40
        self.red_sec = 5
        self.max_phase = 4
        self.last_change_step = {}
        self.agent_list = []
        self.phase_passablelane = {}

        self.memory = deque(maxlen=2000)
        self.learning_start = 2000
        self.update_model_freq = 1
        self.update_target_model_freq = 20

        self.gamma = 0.95  # discount rate
        self.epsilon = 0.1  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.005
        self.batch_size = 32
        self.ob_length = 24

        self.action_space = 8

        self.model = self._build_model()

        # Remember to uncomment the following lines when submitting, and submit your model file as well.
        path = os.path.split(os.path.realpath(__file__))[0]
        
        # Change this to use different Model.h5
        ##############################
        ##############################
        self.load_model(path, 1) # 1->99
        ##############################
        ##############################

        self.target_model = self._build_model()
        self.update_target_network()



    ################################
    # don't modify this function.
    # agent_list is a list of agent_id
    def load_agent_list(self,agent_list):
        self.agent_list = agent_list
        self.now_phase = dict.fromkeys(self.agent_list,1)
        self.last_change_step = dict.fromkeys(self.agent_list,0)
    # intersections[key_id] = {
    #     'have_signal': bool,
    #     'end_roads': list of road_id. Roads that end at this intersection. The order is random.
    #     'start_roads': list of road_id. Roads that start at this intersection. The order is random.
    #     'lanes': list, contains the lane_id in. The order is explained in Docs.
    # }
    # roads[road_id] = {
    #     'start_inter':int. Start intersection_id.
    #     'end_inter':int. End intersection_id.
    #     'length': float. Road length.
    #     'speed_limit': float. Road speed limit.
    #     'num_lanes': int. Number of lanes in this road.
    #     'inverse_road':  Road_id of inverse_road.
    #     'lanes': dict. roads[road_id]['lanes'][lane_id] = list of 3 int value. Contains the Steerability of lanes.
    #               lane_id is road_id*100 + 0/1/2... For example, if road 9 have 3 lanes, then their id are 900, 901, 902
    # }
    # agents[agent_id] = list of length 8. contains the inroad0_id, inroad1_id, inroad2_id,inroad3_id, outroad0_id, outroad1_id, outroad2_id, outroad3_id
    def load_roadnet(self,intersections, roads, agents):
        self.intersections = intersections
        self.roads = roads
        self.agents = agents

        # iterate roads to build a graph
        ''' roads[1000]
         1000: {'start_inter': 12558370237,
                'end_inter': 12997485535,
                'length': 713.0,
                'speed_limit': 22.22222222222222,
                'num_lanes': 3,
                'inverse_road': 999,
                'lanes': {100000: [1, 0, 0], 100001: [0, 1, 0], 100002: [0, 0, 1]}},
                ...}
        '''
        # build traffic graph
        self.traffic_graph = {} # ['inter_from]: [inter_to_1: {speed: xx, length: xx}, inter_to_2, ...]
        for _, road_info in roads.items():
            start_inter_id = int(road_info['start_inter'])
            if (start_inter_id not in self.traffic_graph.keys()):
                self.traffic_graph[start_inter_id] = [] # list of neighbours
            end_inter = {'id': int(road_info['end_inter']), 
                        'length': float(road_info['length']), 
                        'sp_lim': float(road_info['speed_limit']),
                        'num_lanes': int(road_info['num_lanes'])}
            self.traffic_graph[start_inter_id].append(end_inter)


    ################################

    def build_neighbour_features(self, observations_for_agent, feature_name):

        #iterate roads to build the graph
        for inter_id in observations_for_agent.keys():
            neighbours = []
            if inter_id in self.traffic_graph.keys(): # why some agent_id is not in the graph??
                for inter in self.traffic_graph[inter_id]:
                    '''
                    [{'id': 41704581960, 'length': 1016.0, 'sp_lim': 16.666666666666668},
                    {'id': 12296635651, 'length': 103.0, 'sp_lim': 16.666666666666668},
                    {'id': 11356953830, 'length': 349.0, 'sp_lim': 16.666666666666668},
                    {'id': 32296634845, 'length': 43.0, 'sp_lim': 2.7777777777777777}]
                    '''
                    if inter['id'] in observations_for_agent.keys():
                        neighbours.append(
                            {'lane': observations_for_agent[inter['id']][feature_name],
                            'length': inter['length'], 
                            'sp_lim': inter['sp_lim'],
                            'num_lanes': inter['num_lanes']}
                        )
            observations_for_agent[inter_id]['neighbours'] = neighbours
        return observations_for_agent


    def act_(self, observations_for_agent):
        # Instead of override, We use another act_() function for training,
        # while keep the original act() function for evaluation unchanged.

        observations_for_agent = self.build_neighbour_features(observations_for_agent, feature_name='lane')

        actions = {}
        for agent_id in self.agent_list:
            action = self.get_action(observations_for_agent[agent_id]['lane'], observations_for_agent[agent_id]['neighbours'])
            actions[agent_id] = action
        return actions

    def act(self, obs):
        observations = obs['observations']
        info = obs['info']
        actions = {}

        # Get state
        observations_for_agent = {}
        for key,val in observations.items():
            observations_agent_id = int(key.split('_')[0])
            observations_feature = key[key.find('_')+1:]
            if(observations_agent_id not in observations_for_agent.keys()):
                observations_for_agent[observations_agent_id] = {}
            observations_for_agent[observations_agent_id][observations_feature] = val[1:]

        observations_for_agent = self.build_neighbour_features(observations_for_agent, feature_name='lane_vehicle_num')

        # Get actions
        for agent in self.agent_list:
            self.epsilon = 0
            ob_self = observations_for_agent[agent]['lane_vehicle_num']
            ob_msg = observations_for_agent[agent]['neighbours']
            actions[agent] = self.get_action(ob_self, ob_msg) + 1

        return actions

    def aggregate_msg(self, ob_self, ob_msg):
        '''
        [{'lane': [..24...], 'length': 1016.0, 'sp_lim': 16.666666666666668},
        {'lane': [..24...], 'length': 103.0, 'sp_lim': 16.666666666666668},
        {'lane': [..24...], 'length': 349.0, 'sp_lim': 16.666666666666668}]
        '''
        # loop over lane feature, sum with sp_lim/length as weights
        aggregated_msg = 1.0 * np.array(ob_self)
        for inter_feature in ob_msg:
            alpha = inter_feature['num_lanes'] * inter_feature['sp_lim'] / inter_feature['length']
            aggregated_msg += ((alpha * np.array(inter_feature['lane'])) / len(ob_msg)) # normalized weighted sum
        
        return list(aggregated_msg)


    def get_action(self, ob_self, ob_msg): # self is self feature,m msg is feature from neighbours

        # The epsilon-greedy action selector.

        if np.random.rand() <= self.epsilon:
            return self.sample()

        ob_self = self._reshape_ob(ob_self)
        ob_msg = self._reshape_ob(self.aggregate_msg(ob_self, ob_msg))

        act_values = self.model.predict([ob_self, ob_msg])
        return np.argmax(act_values[0])

    def sample(self):

        # Random samples

        return np.random.randint(0, self.action_space)

    def _build_model(self):

        # Neural Net for Deep-Q learning Model

        skip = Input(shape=(self.ob_length, ))
        skip_dense = Dense(20, activation='relu')(skip)

        msg = Input(shape=(self.ob_length, ))
        msg_dense = Dense(20, activation='relu')(msg)

        merge = concatenate([skip_dense, msg_dense])
        out = Dense(self.action_space, activation='linear')(merge)

        model = Model(inputs=[skip, msg], outputs=out)
        
        model.compile(
            loss='mse',
            optimizer=RMSprop()
        )

        return model

    def _reshape_ob(self, ob):
        return np.reshape(ob, (1, -1))

    def update_target_network(self):
        weights = self.model.get_weights()
        self.target_model.set_weights(weights)

    def remember(self, ob, action, reward, next_ob):
        self.memory.append((ob, action, reward, next_ob))

    def replay(self):
        # Update the Q network from the memory buffer.

        if self.batch_size > len(self.memory):
            minibatch = self.memory
        else:
            minibatch = random.sample(self.memory, self.batch_size)
        state, actions, rewards, new_state, = [np.stack(x) for x in np.array(minibatch).T]

        obs_self = state['lane']
        obs_msg = state['neighbours']
        new_obs_self = new_state['lane']
        new_obs_msg = new_state['neighbours']

        target = rewards + self.gamma * np.amax(self.target_model.predict([new_obs_self, new_obs_msg]), axis=1)
        target_f = self.model.predict([obs_self, obs_msg])
        for i, action in enumerate(actions):
            target_f[i][action] = target[i]
        self.model.fit([obs_self, obs_msg], target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load_model(self, dir="model/dqn", step=0):
        name = "dqn_agent_{}.h5".format(step)
        model_name = os.path.join(dir, name)
        print("load from " + model_name)
        self.model.load_weights(model_name)

    def save_model(self, dir="model/dqn", step=0):
        name = "dqn_agent_{}.h5".format(step)
        model_name = os.path.join(dir, name)
        self.model.save_weights(model_name)

scenario_dirs = [
    "test"
]

agent_specs = dict.fromkeys(scenario_dirs, None)
for i, k in enumerate(scenario_dirs):
    # initialize an AgentSpec instance with configuration
    agent_specs[k] = GraphAgent()
    # **important**: assign policy builder to your agent spec
    # NOTE: the policy builder must be a callable function which returns an instance of `AgentPolicy`

