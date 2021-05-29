# CityBrainChallenge-starter-kit

### Useful commands
__evaluate__
```
python3 evaluate.py --input_dir agent-graph --output_dir out/graph --sim_cfg cfg/simulator.cfg --metric_period 200
```
__train__
```
python3 train_dqn_graph.py --input_dir agent-graph --sim_cfg ./cfg/simulator.cfg --output_dir out/graph --episodes 100 --save_model --save_dir agent-graph --save_rate 10 
```

/usr/local/cuda/lib:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:/opt/amazon/efa/lib:/opt/amazon/openmpi/lib:/usr/local/lib:/usr/lib:/usr/lib/x86_64-linux-gnu:


docker run -it -v /home/ubuntu/KDDCup2021-CityBrainChallenge-starter-kit:/starter-kit -v /usr/local/cuda:/usr/local/cuda/lib -v /usr/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu -v /usr/local/cuda/lib64:/usr/local/cuda/lib64 -v /usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda/extras/CUPTI/lib64 citybrainchallenge/cbengine:0.1.2 bash 
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib:/usr/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64

/usr/local/cuda/lib:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:/opt/amazon/efa/lib:/opt/amazon/openmpi/lib:/usr/local/lib:/usr/lib:/usr/lib/x86_64-linux-gnu