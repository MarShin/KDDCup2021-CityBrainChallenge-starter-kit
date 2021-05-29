# CityBrainChallenge-starter-kit

### Useful commands
__evaluate__
```
python3 evaluate.py --input_dir agent-graph --output_dir out/graph --sim_cfg cfg/simulator.cfg --metric_period 200
```
__train__
```
python3 train_dqn_graph.py --input_dir agent-graph --sim_cfg ./cfg/simulator.cfg --output_dir out/graph --episodes 2 --save_model --save_dir agent-graph --save_rate 10 
```