# CHANGELOG

## [0.5.7] 2021-07-06

### Fixed

- Change `start_in_frame` and `end_in_frame`
- Change `latticerandomwalk` to `randomwalk`

### Removed

- Remove `max_trajectories` param from plots

## [0.5.6] 2021-07-02

### Added

- Add example 5 to docs

### Fixed

- Fix langevin generator time array
- Fix default unit on trajectory plots
- Change lattice generator step_length parameter
- Fix trajectory threshold

## [0.5.5] 2021-06-24

### Added

- Add example 4 in docs

### Fixed

- Fix unconsistent trajectory creation
- Fix list of contents in examples

## [0.5.4] 2021-06-21

### Added

- Add rotate function to trajectories

## [0.5.3] 2021-06-18

### Added

- Add example 3 to docs
- Add trajectory slicing

### Fixed

- Fix a bug in time arrays from tracking

## [0.5.1] 2021-06-17

### Added

- Add `uniformly_spaced` property on trajectory
- Add constant add and sub on trajectories
- Add contant multiplication on trajectories

### Fixed

- Change time data calculation if only dt is given
- Fix comparation using threshold
- Fix a bug on default parameters of trackers
- Fix logging on algorithms

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

