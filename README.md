<h1 align="center">Solving multiphysics-based inverse problems with learned surrogates and constraints</h1>

[![][license-img]][license-status] [![][zenodo-img]][zenodo-status]

Code to reproduce results in Ziyi Yin, Rafael Orozco, Mathias Louboutin, Felix J. Herrmann, "[Solving multiphysics-based inverse problems with learned surrogates and constraints](https://arxiv.org/abs/2307.11099)". Currently under minor revision at Advanced Modeling and Simulation in Engineering Sciences.

## Software descriptions

All of the software packages used in this paper are fully *open source, scalable, interoperable, and differentiable*. The readers are welcome to learn about our software design principles from [this open-access article](https://library.seg.org/doi/10.1190/tle42070474.1).

#### Wave

We use [JUDI.jl](https://github.com/slimgroup/JUDI.jl) for wave modeling and inversion, which calls the highly optimized propagators of [Devito](https://www.devitoproject.org/).

#### Multiphase flow

We use [JutulDarcyRules.jl] to solve the multiphase flow equations, which calls the high-performant and auto-differentiable numerical solvers in [Jutul.jl] and [JutulDarcy.jl]. [JutulDarcyRules.jl] is designed to interoperate these two packages with other Julia packages in the Julia AD ecosystem via [ChainRules.jl].

#### Scientific machine learning

We use [InvertibleNetworks.jl] to train the normalizing flows (NFs). This package implements memory-efficient invertible networks via hand-written derivatives. This ensures that these invertible networks are scalable to realistic 3D problems.

We use [FNO4CO2.jl] to train the Fourier neural operators (FNOs) as learned surrogates for multiphase flow solvers in [JutulDarcyRules.jl]. In order for scaling to realistic 4D problems, we suggest the readers also have a look at the [dfno] package, which implements [model-parallel Fourier neural operators based on domain decomposition](https://doi.org/10.1016/j.cageo.2023.105402).

## Installation

First, install [Julia](https://julialang.org/) and [Python](https://www.python.org/). Next, run the command below to install the required packages.

```bash
julia -e 'Pkg.add("DrWatson.jl")'
julia --project -e 'using Pkg; Pkg.instantiate()'
```

## Scripts

There are 12 scripts for permeability inversion with different setups and different types of measurements, listed below.

| Inversion method \ Measurement type | Well measurement | Time-lapse seismic | Both |
|---------------------|----------|----------|--------|
| Unconstrained inversion with PDE solvers | [well-jutul.jl](scripts/well-jutul.jl) | [seismic-jutul.jl](scripts/seismic-jutul.jl) | [combine-jutul.jl](scripts/combine-jutul.jl) |
| Unconstrained inversion with FNO surrogates | [well-fno.jl](scripts/well-fno.jl) | [seismic-fno.jl](scripts/seismic-fno.jl) | [combine-fno.jl](scripts/combine-fno.jl) |
| Constrained inversion with PDE solvers | [well-jutul-nf.jl](scripts/well-jutul-nf.jl) | [seismic-jutul-nf.jl](scripts/seismic-jutul-nf.jl) | [combine-jutul-nf.jl](scripts/combine-jutul-nf.jl) |
| Constrained inversion with FNO surrogates | [well-fno-nf.jl](scripts/well-fno-nf.jl) | [seismic-fno-nf.jl](scripts/seismic-fno-nf.jl) | [combine-fno-nf.jl](scripts/combine-fno-nf.jl) |

Also, the script [projection-study.jl](scripts/projection-study.jl) demonstrates a case study of the relationship between projection in the latent space (expansion or shrinkage) and the accuracy of pre-trained FNO for both in-distribution and out-of-distribution permeability samples.

## LICENSE

The software used in this repository can be modified and redistributed according to [MIT license](LICENSE).

## Reference

If you use our software for your research, we appreciate it if you cite us following the bibtex in [CITATION.bib](CITATION.bib).

## Authors

This repository is written by [Ziyi Yin] from the [Seismic Laboratory for Imaging and Modeling] (SLIM) at the Georgia Institute of Technology.

If you have any question, we welcome your contributions to our software by opening issue or pull request.

SLIM Group @ Georgia Institute of Technology, [https://slim.gatech.edu](https://slim.gatech.edu/).      
SLIM public GitHub account, [https://github.com/slimgroup](https://github.com/slimgroup).    

[Jutul.jl]:https://github.com/sintefmath/Jutul.jl
[JutulDarcy.jl]:https://github.com/sintefmath/JutulDarcy.jl
[JutulDarcyRules.jl]:https://github.com/slimgroup/JutulDarcyRules.jl
[ChainRules.jl]:https://github.com/JuliaDiff/ChainRules.jl
[license-status]:LICENSE
[license-img]:http://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat?style=plastic
[Seismic Laboratory for Imaging and Modeling]:https://slim.gatech.edu/
[FNO4CO2.jl]:https://github.com/slimgroup/FNO4CO2
[InvertibleNetworks.jl]:https://github.com/slimgroup/InvertibleNetworks.jl
[dfno]:https://github.com/slimgroup/dfno
[Ziyi Yin]:https://ziyiyin97.github.io/
[zenodo-status]:https://doi.org/10.5281/zenodo.8353547
[zenodo-img]:https://zenodo.org/badge/DOI/10.5281/zenodo.8353547.svg?style=plastic
