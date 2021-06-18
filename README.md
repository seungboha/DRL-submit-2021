# Active Learning by RL
Run build_agent.py to run the example I made.
This is the step to train the agent.
Unfortunately, I couldn't test the AL agent since trainig took a lots of time. 

Requirement : 

    - numpy
    - sklearn
    - tqdm
    - tensorflow == 1.x
    - matplotlib
    - pickle

Environment :

    - anaconda env
    - linux 

Reference : Discovering General-Purposed Active Learning Strategies (Konyushkova et al. 2018)
https://github.com/ksenia-konyushkova/LAL-RL

To run the example that the paper represented, do the following :
Use 'build agent example.ipynb' to train an AL agent with reinforcement learning and use 
'test agent example.ipynb' to run an AL experiment to test an agent and other baselines. 
The agent will be stored in ./agents and the results of AL experiment will be stored in 
./AL_results.

