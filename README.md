# hessian_of_permanent_is_full_rank_demonstration

# About
This is a code repository for creating an html slide presentation that documents the series of matrix reduction operations used in the paper to show that the truncated Hessian of the permanent of the specially chosen matrix has full rank.

# Setup
This code has been tested with Python 3.9. After cloning the repository into a local directory, changing into that directory (and optionaly creating a new virtual environment) install the dependencies:
```
pip install -r requirements.txt
```

# Use
Run the program by 
```
python presentation.py
```
The output will be a self contained html file in the current working directory that can be opened directly by a web browser. The default name of this file is `output-<d>.html`, where `d` is the dimension argument passed to `presentation.py`. The default value of `d` is `4`.

To see a list of the command line arguments that can be passed to `presentation.py` as well as their default values run
```
python presentation.py --help
```
