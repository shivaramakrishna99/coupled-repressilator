# Implementation for a 6-gene coupled repressilator

#### Current: Toggle Switch and Repressilator - Python implementation

This repository contains a Python implementation of a genetic toggle switch and 3-gene repressilator, as seen in Collins et al [1], and Elowitz et al [2], respectively.

The plots in the Python notebooks show oscillations for the 3-gene repressilator, and nullclines for the toggle.

## Installation
We make use of the ```bioircuits``` package, scipy for computation, and Bokeh for visualization. For replicating the Jupyter notebooks, please install all the packages in ```requirements.txt``` using ```pip install -r requirements.txt```.


[1] Gardner TS, Cantor CR, Collins JJ. Construction of a genetic toggle switch in Escherichia coli. Nature. 2000;403(6767):339-342. doi:10.1038/35002131

[2] Elowitz MB, Leibler S. A synthetic oscillatory network of transcriptional regulators. Nature. 2000;403(6767):335-338. doi:10.1038/35002125