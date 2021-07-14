<!-- Short description -->
<p align="center">
   ORSuite: Benchmarking Suite for Sequential Operations Models
</p>

<!-- The badges -->
<p align="center">
  <a href='https://orsuite.readthedocs.io/en/latest/?badge=latest'>
    <img src='https://readthedocs.org/projects/orsuite/badge/?version=latest' alt='Documentation Status' />
  </a>
</p>

<!-- Horizontal rule -->
<hr>

<!-- Table of content -->

# OR Suite
Reinforcement learning (RL) is a natural model for problems involving real-time sequential decision making. In these models, a principal interacts with a system having stochastic transitions and rewards and aims to control the system online (by exploring available actions using real-time feedback) or offline (by exploiting known properties of the system).

These project revolves around providing a unified landscape on scaling reinforcement learning algorithms to operations research domains.

### Link to Documentation
https://orsuite.readthedocs.io/en/latest/

### Installation Guide

In order to install the required dpeendencies for a new conda environment, please run:
```
conda create --name ORSuite python=3.8.5
conda activate ORSuite
python -m pip install -r requirements.txt
```

### High-Level Overview

The repository has three main components as a traditional Reinforcement Learning set-up :
1. Environments : Environment for the agent to interact with and reside in. `~/or_suite/envs`
2. Agents : Choice of Algorithm `~/or_suite/agents`
3. Experiments : This is a take on implementing the enviroment and agents with a choice of algorithm `~/or_suite/experiment`

### Contribution Guide

See 'ORSuite Contribution Guide' to see information on how to add new environments and algorithms to the package.
