# Biologically-Inspired Neuronal Adaptation Improves Learning in Neural Networks
This is code to reproduce our results from manuscript: "Biologically-Inspired Neuronal Adaptation Improves Learning in Neural Networks":
http://

To run CHL_clamped_adp.py, go:

```
python CHL_clamped_adp.py 
```
*for this code, please install pytoch.


*This python code will create a directory "results" to save the results (log.txt) and parameters.

Currently, number of epochs is set up to 101 (execution time ~38min with GPU Geforce RTX 2080 Super). You can change number of epochs to 60001 ~ (line #405) for full training. <br/>

*You might get an error in line #313 if you are using sklearn version different than # 0.23.2. If you get an error, you can  comment out the line #313, but training data will not be shuffled between epochs.
