<p align="center">
  <img src="https://github.com/micheleFraccaroli/Symbolic_DNN-Tuner/blob/ai@edge/logo/DNN-Tuner_icon.png?raw=true" width=450>
</p>

# EDGE-DNN Tuner
**EDGE-DNN Tuner** is a modular Hardware-Aware NAS framework that extends [Symbolic DNN-Tuner](https://github.com/micheleFraccaroli/Symbolic_DNN-Tuner) 
to optimize deep learning models for edge deployment. It balances network accuracy with strict physical constraints such as latency, FLOPs, and parameter counts. 
Featuring a highly customizable architecture, it allows users to integrate proprietary modules and custom hardware constraints, 
enabling efficient hardware-software co-design through either on-device profiling or analytical cost models.
It supports multiple optimization strategies, including Random Search (RS), Bayesian Optimization (BO), and rule-filtered variants (RS+Rules, BO+Rules) 
that discard invalid architectures early. Crucially, it includes the original tuner's advanced BO+Rules with restart mechanism, which automatically 
resets the search if structural rules conflict with the BO's internal memory.

Full documentation in both English and Italian is available within the Docs folder.
#

