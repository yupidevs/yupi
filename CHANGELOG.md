# CHANGELOG

## [0.3.0]

### Added

* Add getting started video
* Add cross refs to api reference
* Add plot angle distribution
* Add kurtosis and msd plots
* Add plots of velocity autocorrelation function
* Add a function to read and write multiple trajectories
* Add properties doc on trajectory class
* Add new docs on Trajectory class
* Add analizing sections in doccumentation

### Fixed

* Update docs
* Update readme
* Refactor subsample trajectory
* Refactor wrap_theta function
* Refactor diff and velocities funcs
* Refactor estimation of turning angles
* Refactor MSD
* Refactor kurtosis
* Refactor vacf
* Update Langevin Generator to handle scales
* Fix contruction of Trajectory objects in Generators
* Fix t in Trajectories in Generator for Random Walker
* Rename RandomWalker to LatticeRandomWalker
* Refactor velocity histogram function
* Update plot_trajectory parameters
* Refactor trajectory method names
* Refactor Trajectory class attributes handling
* Update docs with a simpler Generator example

### Removed

* Remove old video from doc sources
* Remove unnecessary imports on trajectory

## [0.2.4]

### Fixed

* Fix trajectory array names

## [0.2.3]

### Added

* Add section About in docs
* Add Getting Started section in docs
* Add basic trajectory docstring

### Fixed

* Fix type hints on docs
* Fix docs api reference structure
* Update LE class
* Fix trajectory load json files
* Refactor generating submodule
* Convert trajectory arrays to np.ndarrays

## [0.2.2]

### Added

* Create methods to set initial conditions for position and velocity vectors.
* Create method to solve the stochastic differential equation using the numerical method of Euler-Maruyama.
* Add preprocessing to object tracker
* Add some docs to tracking scenario
* Add method to include noise properties and noise array as atributes of the LE class.
* Method to compute position vectors in RandomWalk class.
* Set constructor for LE class.
* Set constructor for RandomWalk class.

### Fixed

* Fix object tracker preprocessing type hint
* Fix a bug in the function to change color space
* Update ROI size default value to 1
* Refactor walkers into a generation folder
* Update docs

## [0.2.1]

### Added

* Add scale to tracking scenario

### Fixed

* Fix multiple trajectories plot
* Fix roi manual selection

## [0.2.0]

### Added

* Add loggin
* Create LICENSE
* Add color matching algorithm
