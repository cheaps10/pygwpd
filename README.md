===============================
PYthon Gaussian WavePacket Dynamics
PYGWPD
===============================

April 2018

If you use this code, please cite the following two papers:
Heaps, C. W. & Mazziotti, D. A. *The Journal of Chemical Physics*, **144**, 164108 (2016). Pseudospectral Gaussian quantum dynamics: Efficient sampling of potential energy surfaces. https://aip.scitation.org/doi/10.1063/1.4946807

Heaps, C. W. & Mazziotti, D. A. *The Journal of Chemical Physics*, **145**, 064101 (2016). Accurate non-adiabatic quantum dynamics from pseudospectral sampling of time-dependent Gaussian basis sets. https://aip.scitation.org/doi/citedby/10.1063/1.4959872

PYGWPD is a pure python package for performing quantum molecular dynamics simulations on model potential energy surfaces (PES). The focus of the package is Pseudospectral Gaussian Quantum Dynamics, the subject of my dissertation.  

Much of the code in the repository started from graduate school and was a few years ago (pre-2016). Some parts have been heavily revised to be more user-friendly while others are relatively untouched. As a result, there may be some bugs in some options and there might be some bizarre relics in the code.  There are also no guarantees with respect to bugs or accuracy. The two core components, adiabatic and non-adiabatic dynamics, have a lot of similar code due to the nature of the research.  Ideally, the code would be refactored, but I have not had time.

I hope the code may serve as an educational and illustrative tool for Gaussian wave packet dynamics. For me, writing it was an educational experience in Python and quantum dynamics and I'll be happy if it helps others.  

* Quick Start

As with so much code developed by graduate students, there is no real documentation and comments are present but at times unreliable.

I have included a Jupyter Notebook with some heavily commented examples that walk you through running calculations.  A few are available as standalone python script input files, too.


* Capabilities
1. Pseudospectral Gaussian Dynamics
    - Calculates time-derivative of QM basis set coefficients using a pseudospectral method that projects the test space on to basis function centers
    - Perform single-surface (adiabatic) quantum molecular dynamics (QMD) on model PESs including Morse potential and the n-dimensional Henon-Heiles up to 6-D
    - Perform nonadiabatic QMD in both diabatic and adiabatic representations on model PESs inclding Tully models and coupled Morse potentials 
    - Propagates basis set of independent Gaussians using classical equations of motion for adiabatic dynamics and Ehrenfest trajectories for nonadiabatic dynamics.
2. Spectral (Galerkin) Gaussian Dynamics
    - Calculates time-derivative of QM basis set coefficients using the Galerkin method, which calculates the inner product between the basis functions to generate releavant matrices.
    - Analytical PES integrals available for Morse and Henon-Heiles.  The bra-ket averaged Taylor expansion (BAT) is availible for all coded surfaces, allowing comparison of the two.
    - Only adiabatic dynamics are currently available.  The extension should be feasible between the nonadiabatic pseudospectral code + the single-surface Galerkin, but it is not in place.
3. Sinc pseudospectral method
    - The sinc pseudospectral is used to generate reference data for 1 and 2-d cases.

* Restrictions
1. Nonadiabatic calculations in the adiabatic representation
    - Since model surfaces are defined in the diabatic representation, we need to transform them to the adiabatic basis (diagonal in potential energy).  The only code I have for this is from very early on and only valid for 2-surface cases and (maybe, but need to test) 1-dimensional cases
    
* Prerequistes
    - Python 3
    - Numpy
    - Scipy

I encourage having an Anaconda Python 3 environment and you should be fine.  

* Running calculations
The only requirement to run calculations outside of the source directory is to add the pygwpd directory to your python path.  You can add to your .bashrc or .bash\_profile  

export PYTHONPATH=/path_to_pygwpd/pygwpd:$PYTHONPATH

and then you can run calculations in a separate directory.

* Input files
    - A few example input files are included.  They are just python scripts that set run parameters, build the class, and propagate.  By default, time correlation functions and populations (for nonadiabatic caulations) are saved to text files.  For 1 and 2-dimensional cases you can calculate the WF on a grid at times in wf\_times, if you would like to see the actual wavepacket.
    - The included Jupyter Notebook offers some illustrative examples and a visualization of wave packet dynamics.
