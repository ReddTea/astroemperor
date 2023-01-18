# Reddutils

This is ReddTea's small package compiling useful self-made functions for an astrophysical context.

# Overview
Reddutils contains many functions and classes for personal use.
Some of them may be of use for the community.

# Dependencies

This code makes use of:
  - Numpy
  - pandas
  - Scipy
  - PyAstronomy (http://www.hs.uni-hamburg.de/DE/Ins/Per/Czesla/PyA/PyA/index.html)
  - tqdm (https://pypi.python.org/pypi/tqdm)
  - tabulate
  - matplotlib.pyplot
  - ipywidgets
  - IPython.display
  
Most of them come with conda, if some are missing they can be easily installed with pip.

# Installation

In the console type in your work folder
```sh
pip install reddutils
```

# Usage

## Correlator
For a pandas dataframe or three-column table:
```sh
from reddutils import correlator as rc
import pandas as pd

df = pd.read_csv('data.csv')

cor = rc.correlator()
cor.display(df)
```
![correlator](https://user-images.githubusercontent.com/14165443/185804943-5011ccb0-2dff-4d20-a443-8407518b96e5.png)


## Exodus
Use for visualising the NASA Exoplanet Archive:

```sh
from reddutils import exodus

#exopop = exodus.Exoplanet_Archive()
exopop = exodus.Exoplanet_Archive('NasaExoplanetArchive')
#exopop = exodus.Exoplanet_Archive('ExoplanetEU2')
#exopop = exodus.Exoplanet_Archive('ExoplanetsOrg')

exopop.display()
```
![exodus](https://user-images.githubusercontent.com/14165443/185804945-e6753a05-5750-4d00-96fe-657825c39fa1.png)

Try out the histogram mode:

```sh
exopop.display_hist()
```

## Fourier Transform Visualiser
For visualising fourier transforms and the nyquist limit.

```sh
from reddutils import fourier

fou = fourier.fourier()
fou.display()
```
![fourier](https://user-images.githubusercontent.com/14165443/185804946-50b5661a-3a1e-46b7-92a6-a2dcf0d489cd.png)


## Periodogram
For a pandas dataframe:
```sh
from reddutils import periodogram as rp
import pandas as pd

df = pd.read_csv('data.csv')

per = rp.LSP()
per.display(df)
```
![periodogram](https://user-images.githubusercontent.com/14165443/185804947-ce3b2e0e-2019-424d-9f17-bcb3a3ad144d.png)


