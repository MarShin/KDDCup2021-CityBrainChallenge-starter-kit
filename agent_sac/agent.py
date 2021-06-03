""" Required submission file

In this file, you should implement your `AgentSpec` instance, and **must** name it as `agent_spec`.
As an example, this file offers a standard implementation.

Soft-Actor-Critic referring from OpenAI spinningup on optimie trick
niave entropy - fixed, no tuning
"""

import sys
import random
from pathlib import Path
import os
from collections import deque
import os
import itertools

import torch as T
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from buffer import ReplayBuffer
from networks import ActorNetwork, CriticNetwork


path = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(path)

# contains all of the intersections


class SACAgent:
    def __init__(self):

        # CBEngine params
        self.now_phase = {}
        self.green_sec = 40
        self.red_sec = 5
        self.max_phase = 4
        self.last_change_step = {}
        self.agent_list = []
        self.phase_passablelane = {}

        # SAC params
        self.memory = None
        self.learning_start = 2000
        self.update_model_freq = 1
        self.update_target_model_freq = 20

        self.act_dims = 8  # action_space
        self.input_dims = 24  # observation space
        self.ob_length = self.input_dims # align with eval script

        # Remember to uncomment the following lines when submitting, and submit your model file as well.
        # TODO: at inference time set_params() and buid_models() are replaced and set in __init__
        # path = os.path.split(os.path.realpath(__file__))[0]
        # self.load_model(path, 99)
        # self.target_model = self._build_model()
        # self.update_target_network()

    def set_params(self, config):
        # self.act_limit = env.action_space.high[0]
        self.alpha = config["alpha"]  # ideally auto learn temperature instead of fixed
        self.gamma = config["gamma"]  # discount rate
        self.max_size = config["max_size"]
        self.tau = config["tau"]
        self.lr = config["lr"]
        self.layer1_size = config["layer1_size"]
        self.layer2_size = config["layer2_size"]
        self.batch_size = config["batch_size"]
        self.memory = ReplayBuffer(self.max_size, self.input_dims, self.act_dims)

    def build_models(self):
        self.actor = ActorNetwork(
            lr=self.lr,
            input_dims=self.input_dims,
            act_dims=self.act_dims,
            act_limit=self.act_limit,
            name="actor",
        )

        self.critic_1 = CriticNetwork(
            lr=self.lr,
            input_dims=self.input_dims,
            act_dims=self.act_dims,
            name="critic_1",
        )
        self.critic_2 = CriticNetwork(
            lr=self.lr,
            input_dims=self.input_dims,
            act_dims=self.act_dims,
            name="critic_2",
        )
        self.target_critic_1 = CriticNetwork(
            lr=self.lr,
            input_dims=self.input_dims,
            act_dims=self.act_dims,
            name="target_critic_1",
        )
        self.target_critic_2 = CriticNetwork(
            lr=self.lr,
            input_dims=self.input_dims,
            act_dims=self.act_dims,
            name="target_critic_2",
        )

    ################################
    # don't modify this function.
    # agent_list is a list of agent_id
    def load_agent_list(self, agent_list):
        self.agent_list = agent_list
        self.now_phase = dict.fromkeys(self.agent_list, 1)
        self.last_change_step = dict.fromkeys(self.agent_list, 0)

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
    def load_roadnet(self, intersections, roads, agents):
        self.intersections = intersections
        self.roads = roads
        self.agents = agents

    ################################

    def act_(self, observations_for_agent):
        # Instead of override, We use another act_() function for training,
        # while keep the original act() function for evaluation unchanged.

        actions = {}
        for agent_id in self.agent_list:
            action = self.get_action(observations_for_agent[agent_id]["lane"])
            actions[agent_id] = action
        return actions

    def act(self, obs):
        observations = obs["observations"]
        info = obs["info"]
        actions = {}

        # Get state
        observations_for_agent = {}
        for key, val in observations.items():
            observations_agent_id = int(key.split("_")[0])
            observations_feature = key[key.find("_") + 1 :]
            if observations_agent_id not in observations_for_agent.keys():
                observations_for_agent[observations_agent_id] = {}
            observations_for_agent[observations_agent_id][observations_feature] = val[
                1:
            ]

        # Get actions
        for agent in self.agent_list:
            self.epsilon = 0
            actions[agent] = (
                self.get_action(observations_for_agent[agent]["lane_vehicle_num"]) + 1
            )

        return actions

    def load_model(self, dir="model/sac", step=0):
        name = "sac_agent_{}.h5".format(step)
        model_name = os.path.join(dir, name)
        print("load from " + model_name)
        self.model.load_weights(model_name)

    def save_model(self, dir="model/sac", step=0):
        name = "sac_agent_{}.h5".format(step)
        model_name = os.path.join(dir, name)
        self.model.save_weights(model_name)


scenario_dirs = ["test"]

agent_specs = dict.fromkeys(scenario_dirs, None)
for i, k in enumerate(scenario_dirs):
    # initialize an AgentSpec instance with configuration
    agent_specs[k] = SACAgent()
    # **important**: assign policy builder to your agent spec
    # NOTE: the policy builder must be a callable function which returns an instance of `AgentPolicy`

