# gridlod-random-perturbations

```
# This repository is part of the project for "Numerical upscaling of wave equations with time-dependent multiscale coefficients":
#   https://github.com/BarbaraV/gridlod-timedependent
# Copyright holder: Bernhard Maier, Barbara Verfürth 
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)
```

This repository provides code for the experiments of the paper "Numerical upscaling of wave equations with time-dependent multiscale coefficients" by Bernhard Maier and Barbara Verfürth. The code is based on the module `gridlod`  which has been developed by Fredrik Hellman and Tim Keil and consists of code for PGLOD.  `gridlod` is  provided as a submodule, the files in the subfolder `timedependent` extend the PGLOD with time stepping and the adaptive update strategy and were written by Barbara Verfürth.

## Setup

This setup works with a Ubuntu system. The following packages are required (tested versions):
 - python (v3.8.5)
 - numpy (v1.17.4)
 - scipy (v1.3.3)
 - scikit-sparse (v0.4.4)
 - matplotlib (v3.1.2)
 
Please see also the README of the `gridlod` submodule for required packages and setup.
After cloning this repo with git, initialize the submodule via

```
git submodule update --init --recursive
```

Now, build and activate a python3 virtual environment with

```
virtualenv -p python3 venv3
. venv3/bin/activate
```

and execute the following commands

```
echo $PWD/gridlod/ > $(python -c 'from distutils.sysconfig import get_python_lib; print(get_python_lib())')/gridlod.pth
```
Now you can use every file from the subfolders in the virtualenv. Install all the required python packages for gridlod. 

## Reproduction of experiments

First of all, change into the folder with the experiments

```
cd timedependent
```

Reproduction of the experiments consists of two steps: (i) generating the data and (ii) visualization.

### Generating the data
To reproduce the data of the experiments run

```
python3 exp1.py
```
for the experiments of Section 5.1;

```
python3 exp2a.py
```
and
```
python3 exp2b.py
```
for the experiments of Section 5.2.

Some results, in particular the update percentages of Section 5.2, will be printed on screen and the data for the figures will be stored as .mat-files.
Note that running these experiments may take quite a while.

### Visualization
To reproduce the figures in Section 5, run

```
python3 plot_results.py
```

All data from the experiments are available and stored in the `data`-folder as .mat-files, so you can skip the step of generating the data to save time.


## Note

This code is meant to illustrate the numerical methods presented in the paper. It is not optimized in any way, in particular no parallelization is implemented. See `gridlod` for some further explanations of parallelization of the standard PGLOD with ipyparallel. 
