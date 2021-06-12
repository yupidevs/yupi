# CHANGELOG

## [0.4.2] 2021-06-12

### Added

- Add documentation for example002
- Implement offset add and sub on trajectoies
- Implement add and sub between trajectories
- Implement polar offset

### Fixed

- Fix examples links on documentation

## [0.4.1] 2021-06-09

### Added

- Add yupi exceptions
- Add recursively parameter in load_folder method
- Add framedifferencing tracking algorithm
- Add vector docstrings
- Add a background estimator for videos
- Add lazy paremeter on trajectory
- Add background subtraction as tracking algoritm
- Add `plot_trajectory` function on `__init__.py`
- Add template matching algorithm
- Add optical flow algorithm

### Fixed

- Fix exception handling on trajectory loading
- Change velocity calculation on trajectory
- Fix docstring on load_folder
- Change scipy dependency version
- Fix start_in_frame param in track trackingscenario
- Fix trackers for algorithms requiring 2 frames
- Fix drifting in trackers failing detecting a frame
- Fix docstring in frame differencing
- Fix docstring of some algorithms
- Fix wrong references in the docs of algorithms
- Fix a typo in readme

### Others

- Update the api of tracking algorithms
- Update the readme
- Update docs trajectory creation methods
- All tracking algorithms to return an empty mask
- Set version 0.4.1

## [0.4.0] 2021-05-7

### Added

- Add genetrators docs
- Add pytest in dev dependencies
- Add scale comprobation in roi initialization
- Add vector class
- Add trajectory point on iteration
- Add standar deviation for time

### Fixed

- Fix trajectory class
- Fix noise array declaration
- Change parameter plot for show on visualization
- Change trajectory construction on generators

### Removed

- Remove \\\\ from estimate turning angles

### Others

- Specify input arguments
- Separate vector class

## [0.3.2] 2021-03-25

### Added

- Add scipy dependency

### Fixed

- Fix statistics imports

## [0.3.0] 2021-03-23

### Added

- Add getting started video
- Add cross refs to api reference
- Add plot angle distribution
- Add kurtosis and msd plots
- Add plots of velocity autocorrelation function
- Add a function to read and write multiple trajectories
- Add properties doc on trajectory class
- Add new docs on Trajectory class
- Add analizing sections in doccumentation

### Fixed

- Update docs
- Update readme
- Refactor subsample trajectory
- Refactor wrap_theta function
- Refactor diff and velocities funcs
- Refactor estimation of turning angles
- Refactor MSD
- Refactor kurtosis
- Refactor vacf
- Update Langevin Generator to handle scales
- Fix contruction of Trajectory objects in Generators
- Fix t in Trajectories in Generator for Random Walker
- Rename RandomWalker to LatticeRandomWalker
- Refactor velocity histogram function
- Update plot_trajectory parameters
- Refactor trajectory method names
- Refactor Trajectory class attributes handling
- Update docs with a simpler Generator example

### Removed

- Remove old video from doc sources
- Remove unnecessary imports on trajectory

## [0.2.4] 2021-03-12

### Fixed

- Fix trajectory array names

## [0.2.3] 2021-03-10

### Added

- Add section About in docs
- Add Getting Started section in docs
- Add basic trajectory docstring

### Fixed

- Fix type hints on docs
- Fix docs api reference structure
- Update LE class
- Fix trajectory load json files
- Refactor generating submodule
- Convert trajectory arrays to np.ndarrays

## [0.2.2] 2021-03-08

### Added

- Create methods to set initial conditions for position and velocity vectors.
- Create method to solve the stochastic differential equation using the numerical method of Euler-Maruyama.
- Add preprocessing to object tracker
- Add some docs to tracking scenario
- Add method to include noise properties and noise array as atributes of the LE class.
- Method to compute position vectors in RandomWalk class.
- Set constructor for LE class.
- Set constructor for RandomWalk class.

### Fixed

- Fix object tracker preprocessing type hint
- Fix a bug in the function to change color space
- Update ROI size default value to 1
- Refactor walkers into a generation folder
- Update docs

## [0.2.1] 2021-03-06

### Added

- Add scale to tracking scenario

### Fixed

- Fix multiple trajectories plot
- Fix roi manual selection

## [0.2.0] 2021-03-04

### Added

- Add loggin
- Create LICENSE
- Add color matching algorithms
