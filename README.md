# MCIMO:Multi-class Classification with Fuzzy-feature Observations
This is the official site for the paper "Multiclass Classification with Fuzzy-feature Observations: Theory and Algorithms"(https://ieeexplore.ieee.org/document/9807681). This work is done by 

- Guangzhi Ma (UTS), Guangzhi.Ma@student.uts.edu.au
- Prof. Jie Lu (UTS), Jie.Lu@uts.edu.au
- Dr. Feng Liu (UTS), Feng.Liu@uts.edu.au
- Dr. Zhen Fang (UTS), Zhen.Fang@uts.edu.au
- A/Prof. Guangquan Zhang (UTS), Guangquan.Zhang@uts.edu.au

This paper has been accepted by IEEE-TCYB.

# Software version
PyTorch 1.9.0. Python version is 3.9.7. CUDA version is 11.2.

These python files require some basic scientific computing python packages, e.g., numpy. I recommend users to install python via Anaconda (python 3.9.7), which can be downloaded from https://www.anaconda.com/distribution/#download-section . If you have installed Anaconda, then you do not need to worry about these basic packages.

After you install anaconda and PyTorch, you can run codes successfully. Good luck!

# Data download
You can download the file datasets.zip to get all datasets used in this paper.

# Code
You can run 
```
python main_sys.py 
```
--> get results on the synthetic dataset.

You can run 
```
python main_per.py 
```
--> get results on the perceptions experiment dataset.

You can run 
```
python main_mush.py 
```
--> get results on the mushroom dataset.

You can run 
```
python main_letter.py 
```
--> get results on the letter recognition dataset.

You can run 
```
python main_wealon.py 
```
--> get results on the London weather dataset.

You can run 
```
python main_weawashing.py 
```
--> get results on the Washington weather dataset.


# Citation
If you are using this code for your own researching, please consider citing
```
@article{GM2022Multiclass,
  author={Ma, Guangzhi and Lu, Jie and Liu, Feng and Fang, Zhen and Zhang, Guangquan},
  journal={IEEE Transactions on Cybernetics}, 
  title={Multiclass Classification With Fuzzy-Feature Observations: Theory and Algorithms}, 
  year={2022},
  pages={1-14},
  doi={10.1109/TCYB.2022.3181193}
}
```

# Acknowledgment
GM, JL, FL, ZF and GZ were supported by the Australian Research Council (ARC) under FL190100149.
