<p align="center">
  <img src="https://github.com/AI-Unife/Symbolic_DNN-Tuner/blob/master_show/logo/edge-DNNTuner_logo.png" width=450>
</p>

# EDGE-DNN Tuner
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20200987.svg)](https://doi.org/10.5281/zenodo.20200987)

**EDGE-DNN Tuner** is a modular Hardware-Aware NAS framework that extends [Symbolic DNN-Tuner](https://github.com/micheleFraccaroli/Symbolic_DNN-Tuner) 
to optimize deep learning models for edge deployment. It balances network accuracy with strict physical constraints such as latency, FLOPs, and parameter counts. 
Featuring a highly customizable architecture, it allows users to integrate proprietary modules and custom hardware constraints, 
enabling efficient hardware-software co-design through either on-device profiling or analytical cost models.
It supports multiple optimization strategies, including Random Search (RS), Bayesian Optimization (BO), and rule-filtered variants (RS+Rules, BO+Rules) 
that discard invalid architectures early. Crucially, it includes the original tuner's advanced BO+Rules with restart mechanism, which automatically 
resets the search if structural rules conflict with the BO's internal memory.

Full documentation in both English and Italian is available within the Docs folder.
#

