# POMDP Solvers

An educational project with modules for creating a POMDP (Partially Observable Markov Decision Process) model, implementing and running POMDP solver algorithms. This package was developed during my bachelor thesis to help study POMDP and its solvers. 

## Installation

Python version >=3.5.2. To install dependencies, simply do following: 

> pip install -r requirements.txt



## How it works

##### POMDP Environment

For easier construction of a POMDP environment, [POMDP File Grammar](http://www.pomdp.org/code/pomdp-file-grammar.html) is used to encode environment dynamics. Examples of environments can be found in the 'environments' folder. You could also create a new one as long as it complies with the POMDP file conversions. 

* RockSample-7x8.POMDP: Semantic explanation can be found [here](https://www.cse.msu.edu/~lifengji/docs/RockSample.pdf)
* Tiger-2D.POMDP: [Standard Tiger Problem](https://www.techfak.uni-bielefeld.de/~skopp/Lehre/STdKI_SS10/POMDP_tutorial.pdf).
* Tiger-3D.POMDP: 3-Door version of the Tiger Problem. The main difference is that the agent now has to choose one of the doors to listen if it doesn't want to open a door. Then the observation is given depending on how far away is the tiger when the agent puts its ear against the door.
* GridWorld.POMDP: 
  * A simple 2D grid environment where the agent can only move left, right or halt at the current position. The rewarding states are at the two end states and the attempt to moving out of the grid edge causes a penalty,  
  * A much more general 2D grid world environment can be generated using /environments/grid_world_maker.py. Check out /environments/grid_world_example.py for how to use it. 



##### POMDP Solvers

This package has implemented PBVI ([Point-Based Value Iteration](http://www.cs.mcgill.ca/~jpineau/files/jpineau-ijcai03.pdf)) and POMCP ([Partially Observable Monte Carlo Planning](https://papers.nips.cc/paper/4031-monte-carlo-planning-in-large-pomdps.pdf)). Variable names follows the notations used in the original paper so a read-through of papers would be encouraged.

Solver algorithms extend the blueprint class 'POMDP' and are managed by the PomdpRunner. The runner class reads algorithm configurations in the 'configs' folder, creates the environment model, and use those elements to create an actual POMDP solver. 



## How to run it

```
usage: main.py [-h] [--env ENV] [--budget BUDGET] [--snapshot SNAPSHOT]
               [--logfile LOGFILE] [--random_prior RANDOM_PRIOR]
               [--max_play MAX_PLAY]
               config

Solve pomdp

positional arguments:
  config                The file name of algorithm configuration (without JSON
                        extension)

optional arguments:
  -h, --help            show this help message and exit
  --env ENV             The name of environment's config file
  --budget BUDGET       The total action budget (defeault to inf)
  --snapshot SNAPSHOT   Whether to snapshot the belief tree after each episode
  --logfile LOGFILE     Logfile path
  --random_prior RANDOM_PRIOR
                        Whether or not to use a randomly generated
                        distribution as prior belief, default to False
  --max_play MAX_PLAY   Maximum number of play steps (episodes)

* Example usage:
> python main.py pomcp --env Tiger-3D.POMDP --budget 10
```



## Improvements

* Use POMDPX instead of POMDP file grammar. [POMDPX](http://bigbird.comp.nus.edu.sg/pmwiki/farm/appl/index.php?n=Main.PomdpXDocumentation) is a much more concise grammar for defining a POMDP environment. 
* PomdpParser is carrying too much responsibility — needs to be refactored.
* Configuration implementation still looks a bit messy. 