

## MCMM Project

### Overview
A Markov State Model estimation and analysis toolbox for time series data inspired by [msmtools](https://github.com/markovmodel/msmtools) and
[PyEMMA](https://github.com/markovmodel/PyEMMA). Written in Python, mostly relying on numpy and scipy.

### Contents
This package provides the following functionality:

1.  Basic data clustering (KMeans/KMeans++, Regspace, DBSCAN)
2.  Estimation of transition matrices from trajectory data 
3.  Analysis functionality (eigenvalues/vectors, transition path theory, aperiodicity, irreducibility) for Markov State Models
4.  Visualization for clustering and analysis

### Requirements:
numpy, msmtools>=1.0, matplotlib, scipy, pandas

### Installation
Call
```python setup.py install```
from project directory. Alternatively, call
```
pip install git+https://github.com/Markov-Schmarkov/mcmm-project
``` 
to install from github.

