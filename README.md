<!-- Logo -->
<p align="center">
   <img src="https://raw.githubusercontent.com/cornell-orie/ORSuite/main/images/ORSuite.svg" width="50%">
</p>

<!-- Short description -->
<p align="center">
   ORSuite: Benchmarking Suite for Sequential Operations Models
</p>

<!-- The badges -->
<p align="center">
  <a href="https://github.com/cornell-orie/ORSuite/actions">
    <img alt="pytest" src="https://github.com/cornell-orie/ORSuite/workflows/Test/badge.svg">
  </a>
  <a href='https://orsuite.readthedocs.io/en/latest/?badge=latest'>
    <img src='https://readthedocs.org/projects/orsuite/badge/?version=latest' alt='Documentation Status' />
  </a>
   <!--
  <a href="https://img.shields.io/pypi/pyversions/ORSuite">
     <img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/ORSuite">
  </a> 
   -->
   <a href="https://github.com/cornell-orie/ORSuite/graphs/contributors">
      <img alt="contributors" src="https://img.shields.io/github/contributors/cornell-orie/ORSuite">
   </a>
   <!--
   <a href="https://img.shields.io/pypi/dm/orsuite">
      <img alt="PyPI - Downloads" src="https://img.shields.io/pypi/dm/orsuite">
   </a> 
   -->
</p>






<!-- Horizontal rule -->
<hr>

<!-- Table of content -->



# OR Suite
Reinforcement learning (RL) is a natural model for problems involving real-time sequential decision making. In these models, a principal interacts with a system having stochastic transitions and rewards and aims to control the system online (by exploring available actions using real-time feedback) or offline (by exploiting known properties of the system).

These project revolves around providing a unified landscape on scaling reinforcement learning algorithms to operations research domains.

### Documentation
https://orsuite.readthedocs.io/en/latest/

### Code Demonstration
https://colab.research.google.com/drive/1oSv8pCwl9efqU4VEHgi8KXNvHiPXi7r1?usp=sharing

### Installation Guide

In order to install the required dpeendencies for a new conda environment, please run:
```
conda create --name ORSuite python=3.8.5
conda activate ORSuite
pip install -e .
```

### High-Level Overview

The repository has three main components as a traditional Reinforcement Learning set-up :
1. Environments : Environment for the agent to interact with and reside in. `~/or_suite/envs`
2. Agents : Choice of Algorithm `~/or_suite/agents`
3. Experiments : This is a take on implementing the enviroment and agents with a choice of algorithm `~/or_suite/experiment`

### Contribution Guide

See 'ORSuite Contribution Guide' to see information on how to add new environments and algorithms to the package.
