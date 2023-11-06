# hessian_of_permanent_is_full_rank_demonstration

# About
This is a code repository for creating an HTML slide presentation that documents a series of matrix reduction operations used in a paper submitted to the journal Information Processing Letters that is currently under review. The sequence of matrix operations is used to show that a particular submatrix of the Hessian of the Permanent of a specially chosen matrix is of full rank. 

The actual HTML slide presentation for the dimension 4 case is contained in the file `output-4.html`, which should open in any web browser and can be downloaded by itself instead of cloning this whole repository and building it from scratch.

# Setup
This code has been tested with Python 3.9. After cloning the repository into a local directory, changing into that directory (and optionally creating a new virtual environment) install the dependencies:
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
