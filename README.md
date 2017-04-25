# 21cmpy
PyCUDA implementation of 21cmFAST: in working progress

21cmFAST ([[1]]) is a semi-numerical simulation package for the epoch of reionization. This PyCUDA based implementation aims to improve the original C-code ([[2]]) through the performance benefit of GPU compute and user-friendliness of the Python API. 


![ES](web/smooth.gif)

## Dependencies

```
numpy, matplotlib, pycuda, cosmolopy
```

[[1]] 21cmFAST: A Fast, Semi-Numerical Simulation of the High-Redshift 21-cm Signal. Andrei Mesinger, Steven Furlanetto, Renyue Cen arXiv:1003.3878
[[2]](https://github.com/andreimesinger/21cmFAST) 21cmFast by Andrei Messinger

[[3]](https://github.com/pritchardjr/tocmfastpy) tocmfastpy by Jonathan Pritchard
