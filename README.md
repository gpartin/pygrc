# pygrc

[![License](https://img.shields.io/github/license/amanmdesai/pygrc)](https://github.com/amanmdesai/pygrc/blob/master/LICENSE.txt)
[![CI Tests](https://github.com/amanmdesai/pygrc/actions/workflows/test.yaml/badge.svg)](https://github.com/amanmdesai/pygrc/actions/workflows/test.yaml)
[![Publish](https://github.com/amanmdesai/pygrc/actions/workflows/publish.yml/badge.svg)](https://github.com/amanmdesai/pygrc/actions/workflows/publish.yml)
[![PyPI Version](https://img.shields.io/pypi/v/pygrc)](https://pypi.org/project/pygrc/)
[![Python Versions](https://img.shields.io/pypi/pyversions/pygrc)](https://pypi.org/project/pygrc/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7546312.svg)](https://doi.org/10.5281/zenodo.7546312)
[![Journal of Brief Ideas](https://img.shields.io/badge/Journal_of_Brief_Ideas-10.5281/zenodo.8092602-red)](https://beta.briefideas.org/ideas/71d1533579769a11cae153150ce6f3ea)

## Author

Aman Desai

##  Description

A package to read and fit SPARC data for Galactic Rotation Curves


Galaxy rotation curves are a class of plots which have velocity of the galactic objects on the y-axis and the x-axis represents their radial distances from the galactic center. 


## Analysis Steps 

We present the pygrc package to simpify testing new models against data as well as to make galaxy rotation curve analysis accessible to undergraduates. At present, this package is designed to read the SPARC Data (in particular the data stored under newton models). We have used pandas for reading data, iminuit for fitting the data and matplotlib for plotting.  The definition of model can be implemented by using functions defined in python language and by sharing these function via calls to pygrc package. The fitting procedure uses the least square approach as implemented in the iminuit package. 

An example workflow to carry out analysis with pygrc is as follows: 

-  Read data from the SPARC Dataset - which gives among other things, the observed rotation velocity and the distance of the object from the center of the galaxy.

```python
df = gr.Reader.read(filepath=<path to the file in double quotation>)
```
- (optional) Make some initial plots for visualization and understanding the SPARC data for the given galaxy.

```python
gr.Plot().overlap(df,"Rad",["Vobs","Vgas","Vbul","Vdisk"],"observed velocity")
```

- User needs to define mass distribution along with useful parameters (for example to define the mass density given in \ref{eq}, use: 

```python
	def mass(r, M0, R0):
    return M0*(1- (1+(r/R0))*np.exp(-r/R0))
```

- User then defines a model whose compatibility with data is to be checked in the following way:

```python
def newton(r, M0, rc, R0,beta):
    G = 4.30e-6  
    m = mass(r,M0, R0)
    f = G*m/r
    return np.sqrt(f)*10e-3
```

- Fit the model with Data. First define the radius variable and its range. Then define the  


```python
r = np.linspace(1e-5,df["Rad"].max(),2000)
```

Then define the variables to fit and the initial values to be used as follows:

```python
m_1=gr.Fit(df["Rad"],df["Vobs"],1.,1.,3,.35,1.)
```

where initial parameters values are in the same order as defined in the function *newton*. In the next step we define the range to scan for the parameters as well as define the error steps and finally define which parameters needs to be scanned and which are taken as fixed.

```python
m_1.fit_lsq(f, [(1e4,None),(1e-1,None),(1,10),(0.1,2),(0.1,2)],df['errV'],\\
[False,False,True,True,True])
```

\end{enumerate}



## Installation

```bash
pip install pygrc
```

## Example Usage

- 1. Import libraries

```python
import math
import numpy as np
import matplotlib.pyplot as plt
import pygrc as gr
```

- 2. Read Sparc Data

```python
df = gr.Reader.read(filepath="/home/amdesai/Physics/data/NGC5055_rotmod.dat")
gr.Plot().overlap(df,"Rad",["Vobs","Vgas","Vbul","Vdisk"],"observed velocity")
df
```

- 3.  Some sample function
```python

def mass(r, M0, R0):
    return M0*(1- (1+(r/R0))*np.exp(-r/R0))

def mond(r, M0, rc, R0,b, beta):
    a = 1.2e-10
    G = 4.300e-6 #parsecs
    m = mass(r,M0, R0)
    f = (G*m/r)*(1 + b*(1+(r/rc)))#*10e-3
    return np.sqrt(f)*10e-3

def newton(r, M0, rc, R0,beta):
    a = 1.2e-10
    G = 4.30e-6 #parsecs
    m = mass(r,M0, R0)
    f = G*m/r
    #f = (G*m/r)*(1/np.sqrt(2)) * np.sqrt(1 + np.sqrt(1 + (r**4) * (2*a/(G*m))**2))
    return np.sqrt(f)*10e-3
```

- 4a. Perform fit  (uses iminuit in the background)

```python
r = np.linspace(1e-5,df['Rad'].max(),2000)

m_1=gr.Fit(df['Rad'],df['Vobs'],1.,1.,3,.35,1.)

m_2=gr.Fit(df['Rad'],df['Vobs'],1.,1.,3,1.)
```

- 4b. Continue:

```python
m1= m_1.fit_lsq(mond, [(1.,1e17),(1.,10.),(2.,5.),(0.1,2),(0.1,2)],df['errV'],[False,False,True,True,True])
m1
```

- 4c. Continue:

```python
m2 = m_2.fit_lsq(newton, [(1.,1e16),(1.,10.),(1,10),(0.1,2)],df['errV'],[False,True,True,True])
m2
```

- 5a. draw plots
```python
m_2.get_profile(m2,'M0')
m_1.draw_contours(m1,['M0','rc'])
```

- 6b.

```python
fig, ax =plt.subplots()
gr.Plot().plot_grc(df,m1,mond,'MOND','NGC',ax)
gr.Plot().plot_grc(df,m2,newton,'Newton','NGC',ax)
plt.savefig('1.pdf')
```


## References:

- SPARC Data can be obtained from here: http://astroweb.cwru.edu/SPARC/
- Mond and Mass function are based on Galaxies 2018, 6(3), 70; https://doi.org/10.3390/galaxies6030070
