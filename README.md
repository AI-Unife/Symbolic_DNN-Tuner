<p align="center">
  <img src="https://github.com/micheleFraccaroli/Symbolic_DNN-Tuner/blob/ai@edge/logo/DNN-Tuner_icon.png?raw=true" width=450>
</p>

# Model Bench Utility
**Model Bench** is a benchmarking API designed to: measure the performance of models, focusing on **latency** and **FLOPs**; **conversion** of models; **fine-tuning** processes.

## Setup and Usage
Follow these steps to get the API running:

1. **Install Dependencies**: Install the required packages based on your Operating System:
   ```bash
   pip install -r requirements.txt
   ```
2. **Configure Paths**: Set the local paths for your dataset in:
./components/gesture_dataset.py:
- Edit dataset_path (line 257)
- Edit cache_dir (line 260)
- Edit cache_dir (line 311)
./components/dataset.py
- Edit root (line 26)
3. **Run the API**: Start the service by executing:
  ```bash
  python model_bench/run.py
  ```

#
# Symbolic DNN-Tuner
Symbolic DNN-Tuner is a system to drive the training of a Deep Neural Network, analysing the performance of each training experiment and automatizing the choice of HPs to obtain a network with better performance.

#
## Publications

<a href="https://link.springer.com/article/10.1007/s10994-021-06097-1">[1]</a> Michele Fraccaroli, Evelina Lamma & Fabrizio Riguzzi (2021): Symbolic DNN-Tuner. Machine Learning, pp. 1–26, doi:10.1007/s10994-021-06097-1. <br>
<a href="https://www.sciencedirect.com/science/article/pii/S2352711021001825">[2]</a> Michele Fraccaroli, Evelina Lamma & Fabrizio Riguzzi (2022): Symbolic DNN-Tuner: A Python and ProbLog-based system for optimizing Deep Neural Networks hyperparameters. SoftwareX 17, p. 100957, doi:10.1016/j.softx.2021.100957.<br>

#
## Bibitex Citations
```
@article{fraccaroli2022symbolic,
  title={Symbolic DNN-tuner},
  author={Fraccaroli, Michele and Lamma, Evelina and Riguzzi, Fabrizio},
  journal={Machine Learning},
  volume={111},
  number={2},
  pages={625--650},
  year={2022},
  publisher={Springer}
}
```

```
@article{fraccaroli2022symbolic,
  title={Symbolic DNN-Tuner: A Python and ProbLog-based system for optimizing Deep Neural Networks hyperparameters},
  author={Fraccaroli, Michele and Lamma, Evelina and Riguzzi, Fabrizio},
  journal={SoftwareX},
  volume={17},
  pages={100957},
  year={2022},
  publisher={Elsevier}
}
```# newTuner
