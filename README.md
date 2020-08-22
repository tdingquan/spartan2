
# spartan2:

**spartan2** is a collection of data mining algorithms on **big graphs** and
**time series**, as graphs and time series are fundamental representations of many key applications 
in a wide range of users' online behaviors (e.g. social media, shopping, Apps), 
finance (e.g. stock tradings, bank transfers), IoT networks (e.g. sensor readings, smart power grid), and healthcare (e.g. electrocardiogram, photoplethysmogram, respiratory inductance plethysmography). 

In practice, we find that thinking graphs and time series as matrices or tensors
can enable us to find efficient (near linear), interpretable, yet accurate solutions in many applications.

Therefore, we want to develop a collectioin of algorithms on graphs and time series based
on tensors (matrix is a 2-mode tensor). In real world, those tensors are sparse, and we
are required to make use of the sparsity to develop efficient algorithms. That is why
we name the package of algorithms as 

**SparTAn**: **Spar**se **T**ensor **An**alytics.

spartan2 is backend of SparTAn.
Everything here is viewed as a tensor (sparse).

### install requires

This project requires Python 3.7 and upper.
We suggest recreating the experimental environment using Anaconda through the following steps.
 
1. Install the appropriate version for Anaconda from here - https://www.anaconda.com/distribution/

2. Create a new conda environment named "spartan"
```bash
conda create -n spartan python=3.7
conda activate spartan
```

3. If you are a normal **USER**, download the package from pip

```bash
pip install -i https://test.pypi.org/simple/ spartan2
```

4. If you are a **DEVELOPER** and want to **contribute** to the project, please
- clone the project from github

```bash
git clone https://github.com/shenghua-liu/spartan2.git
``` 

- Install requirements.
```bash
# [not recommended]# pip install --user --requirement requirements
# using conda tool
conda install --force-reinstall -y --name spartan -c conda-forge --file requirements
```
   or  
```bash
# this may not work in ubuntu 18.04
python setup.py install
```

- Install code in development mode
```bash
# in parent directory of spartan2
pip install -e spartan2
```
- Since you install your package to a location other than the user site-packages directory, you will need to 
add environment variable PYTHONPATH in ~/.bashrc  
```bash
export PYTHONPATH=/<dir to spartan2>/spartan2:$PYTHONPATH
```

   or prepend the path to that directory to your PYTHONPATH environment variable.
```python
import sys
sys.path.append("/<dir to spartan2>/spartan2")
```
   or 
```bash
#find directory of site-packages
python -c 'import site; print(site.getsitepackages())'
```

   and add \<name\>.pth file in your site-packages directory

```
/<dir to spartan2>/spartan2
```



5. run the project demos in jupyter notebook:

- start jupyter notebook
- click to see each jupyter notebook (xxx.ipynb) demo for each algorithm, or see the following guidline.


## Live-tutorials: Table of Contents

**Part 1: Basic**
* [Quick start](./live-tutorials/quick_start.ipynb)


**Part 2: Big Graphs**
* [Graph start](./live-tutorials/graph_start.ipynb)
* [SpokEn](./live-tutorials/SVD_demo.ipynb): an implementation of [EigenSpokes](http://www.cs.cmu.edu/~christos/PUBLICATIONS/pakdd10-eigenspokes.pdf) by SVD.
* [Eaglemine](./live-tutorials/Eaglemine_demo.ipynb)
* [Fraudar](./live-tutorials/Fraudar_demo.ipynb): a wrapper of [Fraudar](https://bhooi.github.io/projects/fraudar/index.html) algorithm.
* [Holoscope](./live-tutorials/Holoscope.ipynb): based on [HoloScope](https://shenghua-liu.github.io/papers/cikm2017-holoscope.pdf)

**Part 3: Time Series**
* [Time Series start](./live-tutorials/timeseries_start.ipynb)
* [Basic operations](./live-tutorials/TimeseriesData_demo%20-%20II.ipynb)
* [Beatlex](./live-tutorials/Beatlex_demo.ipynb): based on [BeatLex](https://shenghua-liu.github.io/papers/pkdd2017-beatlex.pdf)

## References
1. Shenghua Liu, Bryan Hooi, Christos Faloutsos, A Contrast Metric for Fraud Detection in Rich Graphs, IEEE Transactions on Knowledge and Data Engineering (TKDE), Vol 31, Issue 12, Dec. 1 2019, pp. 2235-2248.
1. Shenghua Liu, Bryan Hooi, and Christos Faloutsos, "HoloScope: Topology-and-Spike Aware Fraud Detection," In Proc. of ACM International Conference on Information and Knowledge Management (CIKM), Singapore, 2017, pp.1539-1548.
2. Prakash, B. Aditya, Ashwin Sridharan, Mukund Seshadri, Sridhar Machiraju, and Christos Faloutsos. "Eigenspokes: Surprising patterns and scalable community chipping in large graphs." In Pacific-Asia Conference on Knowledge Discovery and Data Mining, pp. 435-448. Springer, Berlin, Heidelberg, 2010.
3. Wenjie Feng, Shenghua Liu, Christos Faloutsos, Bryan Hooi, Huawei Shen, Xueqi Cheng, EagleMine: Vision-Guided Mining in Large Graphs, ACM SIGKDD 2018, ODD Workshop on Outlier Detection De-constructed, August 20th, London UK.
4. Bryan Hooi, Shenghua Liu, Asim Smailagic, and Christos Faloutsos, “BEATLEX: Summarizing and Forecasting Time Series with Patterns,” The European Conference on Machine Learning & Principles and Practice of Knowledge Discovery in Databases (ECML-PKDD), Skopje, Macedonia, 2017.
5. Hooi, Bryan, Hyun Ah Song, Alex Beutel, Neil Shah, Kijung Shin, and Christos Faloutsos. "Fraudar: Bounding graph fraud in the face of camouflage." In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pp. 895-904. 2016.
