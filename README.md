# Multi-objective Simulated Annealing for Hyper-parameter Optimization in Convolutional Neural Networks
Multi-objective Simulated Annealing for Hyper-parameter Optimization in Convolutional Neural Networks

Ayla Gülcü, Zeki Kuş
Dept. of Computer Science, Fatih Sultan Mehmet University, Istanbul, Turkey

This repository containts code for the paper: [Multi-objective simulated annealing for hyper-parameter optimization in convolutional neural networks](https://peerj.com/articles/cs-338/)

### Overview

In this study, we model a CNN hyper-parameter optimization problem as a bi-criteria optimization problem, where the first objective being the classification accuracy and the second objective being the computational complexity which is measured in terms of the number of floating point operations. For this bi-criteria optimization problem, we develop a Multi-Objective Simulated Annealing (MOSA) algorithm for obtaining high-quality solutions in terms of both objectives. CIFAR-10 is selected as the benchmark dataset, and the MOSA trade-off fronts obtained for this dataset are compared to the fronts generated by a single-objective Simulated Annealing (SA) algorithm with respect to several front evaluation metrics such as generational distance, spacing and spread. The comparison results suggest that the MOSA algorithm is able to search the objective space more effectively than the SA method. For each of these methods, some front solutions are selected for longer training in order to see their actual performance on the original test set. Again, the results state that the MOSA performs better than the SA under multi-objective setting. The performance of the MOSA configurations are also compared to other search generated and human designed state-of-the-art architectures. It is shown that the network configurations generated by the MOSA are not dominated by those architectures, and the proposed method can be of great use when the computational complexity is as important as the test accuracy.

![](https://github.com/zekikus/MOSA-cnn-hyperparams-optimization/blob/master/images/mosa_vs_rs.jpg)

#### Figure: Visual comparison of MOSA and RS search ability in terms of objective space distribution and the Pareto fronts with (A) random seed: 10, (B) random seed: 20, (C) random seed:30.

![](https://github.com/zekikus/MOSA-cnn-hyperparams-optimization/blob/master/images/mosa_vs_sa.jpg)

#### Figure: Visual comparison of MOSA and SA search ability in terms of objective space distribution and the Pareto fronts with (A) random seed: 10, (B) random seed: 20, (C) random seed:30.

![](https://github.com/zekikus/MOSA-cnn-hyperparams-optimization/blob/master/images/results_table.png)

#### Table: Comparison of MOSA architectures to other search generated architectures.