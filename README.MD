# Optorepressilator: A Light-Controllable Synthetic Genetic Oscillator

This repository contains the code and data for implementing and analyzing the optorepressilator, a light-controllable synthetic genetic oscillator in E. coli.

This repository also contains a Python implementation of a genetic toggle switch and 3-gene repressilator, as seen in Collins et al [1], and Elowitz et al [2], respectively.

The plots in the Python notebooks show oscillations for the 3-gene repressilator, and nullclines for the toggle.

## Overview

The optorepressilator is a novel four-node optogenetic network that combines the repressilator circuit with an optogenetic module, enabling precise control of genetic oscillations using light inputs. This system allows for:

- Optical synchronization of a population of genetic oscillators
- Entrainment of oscillations to periodic light pulses 
- Tuning of oscillation frequency and phase

## Contents

- `code/`: Python scripts for simulating optorepressilator dynamics
- `figures/`: Scripts to reproduce key figures from the paper

## Implementations

- Simulation of optorepressilator dynamics using ODEs
- Analysis of population-level and single-cell oscillation data
- Implementations of phase response curves and Arnold tongue analysis
- Visualization of synchronization and entrainment phenomena

## Usage

Detailed usage instructions for running simulations and analyzing data can be found in each subdirectory.

## Requirements

See ```requirements.txt```


# Citations

[1] Cannarsa, M.C., Liguori, F., Pellicciotta, N., Frangipane, G., & Di Leonardo, R. (2024). Light-driven synchronization of optogenetic clocks. eLife, 13:e97754.

[2] Gardner TS, Cantor CR, Collins JJ. Construction of a genetic toggle switch in Escherichia coli. Nature. 2000;403(6767):339-342. doi:10.1038/35002131

[3] Elowitz MB, Leibler S. A synthetic oscillatory network of transcriptional regulators. Nature. 2000;403(6767):335-338. doi:10.1038/35002125