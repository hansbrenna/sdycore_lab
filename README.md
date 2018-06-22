# The Simple Dynamical Core Lab
The Simple Dynamical Core Lab (SDYCORE LAB) is a personal project to explore some numerical algorithms involved in primitive equations models of atmospheric dynamics in a simplified setting.

The primitive equations are a set of nonlinear partial differential equations governing the large scale flow of fluids on a rotating planet. All modern weather prediction and general circulation models have at their core a solver for the primitive equations, with or without extra approximations. This part is called the dynamical core and is the most performance-critical part of the model. 

As a result, all dynamical cores (I could find) are written for maximum performance targeting modern supercomputers and clusters. In addition, the field of dynamical core implementation is more than 50 years old, meaning that the methods applied to the problem in modern cores are quite advanced, making them dificult to understand. Even the cores used in simplified models like PUMA are somewhat difficult to understand since they use the spectral transform method and are fully parallelized.

To begin with, stripping away parallelism, choosing simple numerical methods and writing the programs in an easy to understand way is the main goal. Later, I might consider bringing parallelism back in some versions of the cores I plan to develop. Serial performance has been a consideration, though, and I have used a combination of Numba JIT and Fortran to achieve this.

The first project of SDYCORE LAB is the simplest primitive equations solver I could think of. Using finite difference methods to solve the primitive equations with pressure as the vertical coordinate over a flat surface with no grid staggering in the horizontal. This core, called the isobaric dycore and found in this repository is functioning and in an advanced state of development. I will add a version using centered 2nd order finite differences in space and time written in Python and a version using 4th order centered differences in space and 2nd order in time. This version is written i Python and Fortran. Both cores use explicit leapfrog time-stepping.

Plans for the future is to explore the effects of grid staggering, different formulations of the primitive equations, different vertical coordinate systems and different soultion algorithms. I have already decided that I want to implement a finite difference solver on a staggered grid, a spectral solver and finite difference solver in the vorticity-divergence formulation. Plans for the far future might involve implementing a finite volume solver or some other solver.

Important references, resources and inspirations for this project has been (in order of discovery):
1. Edwards, P. N. A Vast Machine: Computer Models, Climate Data, and the Politics of Global Warming. (MIT Press, 2010). doi:10.1111/j.1541-1338.2011.00522_3.x
2. Chang, J. (ed.) Methods in Computational Physics Volume 17: General Circulation Models of the Atmosphere. (Academic Press, 1977)
3. KASAHARA, A. & WASHINGTON, W. M. NCAR GLOBAL GENERAL CIRCULATION MODEL OF THE ATMOSPHERE. Mon. Weather Rev. 95, 389–402 (1967).
4. WASHINGTON, W. M. & KASAHARA, A. A JANUARY SIMULATION EXPERIMENT WITH THE TWO-LAYER VERSION OF THE NCAR GLOBAL CIRCULATION MODEL. Mon. Weather Rev. 98, 559–580 (1970).
5. Oliger, J. E., Wellck, R. E., Kasahara, A. & Washington, W. M. NCAR GLOBAL CIRCULATION MODEL. (1970). NCAR Technical report
6. Kasahara, A. Various Vertical Coordinate Systems Used for Numerical Weather Prediction. Mon. Weather Rev. (1974). doi:10.1175/1520-0493(1974)102
7. Leith, C. E. Numerical Simulation of the Earth's Atmosphere. In Alder, B., Fernbach, S., Rotenberg, M. (eds.) Methods in Computational Physics Volume 4: Applications in Hydrodynamics. (Academic Press 1965)
8. Williamson, D. L. Linear Stability of Finite-Difference Approximations on a Uniform Latitude-Longitude Grid with Fourier Filtering. Mon. Weather Rev. 104, 31–41 (1976).



