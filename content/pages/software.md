Title: Software
Date: 2019-11-15
Modified: 2019-11-15
Slug: Software
Author: Pete Bunting
Summary: Software Tools and Installation

## Installation

### Conda

If you wish to install RSGISLib, ARCSI and the other tools we make available using the conda Python package manager then this video provides a complete run through of the process. Note, the recommended options and versions have been updated from the video so take those from the following commands:

<iframe width="560" height="315" src="https://www.youtube.com/embed/9HqKLioyAeM" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

I would now recommend that you use the [mamba](https://github.com/mamba-org/mamba)  command, rather than [conda](https://docs.conda.io) for the installation as it is quicker and easier. [Mamba](https://github.com/mamba-org/mamba) is a reimplmentation of conda using C++ rather than python. Therefore, all you need to do is to install mamba into your conda environment (command below) and then replace conda with mamba in all the commands you run.

    :::bash
    conda install mamba -n base -c conda-forge


Commands for create an environment and installing RSGISLib, ARCSI etc.
    
    :::bash
    # Create a New Environment
    conda create -n osgeo-env-v1 python=3.10
    
    # Change to the environment
    source activate osgeo-env-v1
    
    # Install RSGISLib and ARCSI
    mamba install -c conda-forge rsgislib arcsi tuiview
    
    # Install RSGISLib other useful packages:
    mamba install -c conda-forge rsgislib h5py parallel scikit-learn scikit-image matplotlib pandas geopandas statsmodels scipy rasterio shapely networkx sqlalchemy pycurl seaborn numba pip rtree pygal jupyterlab pysal libpysal esda pyyaml netcdf4 xarray fiona psycopg2 ipywidgets contextily cvxopt feather-format pyod xlsxwriter openpyxl SALib tuiview
    
    # Some extras only avilable via pip
    pip install matplotlib_scalebar pysptools


### Docker and Singularity
The easiest way to install our software is through Docker or Singularity, as shown below:
    
    :::bash
    # Pull the docker image to your local system
    docker pull petebunting/au-eoed
    
    # Pull the docker image using singularity
    singularity build au-eoed.sif docker://petebunting/au-eoed

## Software We Maintain

### RSGISLib
[http://www.rsgislib.org](https://www.rsgislib.org)

The remote sensing and GIS software library, tools for processing image and vector datasets using Python.

### ARCSI
[http://remotesensing.info/arcsi](http://remotesensing.info/arcsi)

A set of tools for the automated productions of Analysis Ready Optical Data (ARD). Supports Landsat and Sentinel-2.

### EODataDown
[http://remotesensing.info/eodatadown](http://remotesensing.info/eodatadown)

Software for creating an EO based monitoring system.

### KEALib
[http://www.kealib.org](http://www.kealib.org) 

HDF5 based image file format with GDAL driver.

### SPDLib
[http://www.spdlib.org](http://www.spdlib.org) 

HDF5 based file format for LiDAR data and tools for processing LiDAR datasets.

### pylidar
[http://www.pylidar.org](http://www.pylidar.org) 

Python module for reading, writing and processing LiDAR datasets.

### Tuiview
[http://www.tuiview.org](http://www.tuiview.org)

Lightweight Earth Observation (EO) image viewer. 

## Other Software
### Accessing and Storing Data
* [GDAL](http://www.gdal.org) - Reading and writes pretty much all the file formats we use and provides a set of useful tools for manipulating data.
* [RIOS](http://rioshome.org) - Python module for access image pixel values for analysis.
* [Open Data Cube](https://www.opendatacube.org) - Python module for storing and accessing very large timeseries of raster data.

### Visualisating Data
* [QGIS](https://qgis.org) - Open source graphical GIS software.
* [Cartopy](http://scitools.org.uk/cartopy/) - Python module for making maps, built on Matplotlib.

### Useful Python Modules
* [Python-FMASK](http://pythonfmask.org) - This is a python implementation of the FMask algorithm for cloud masking Landsat 4-8 data and Sentinel-2. This library is used with ARCSI to perform cloud masking.
* [scikit-image](http://scikit-image.org) - Python library for manipulating image data. When linked with RIOS it is very powerful in terms of analysing spatial image data.
* [shapely](http://shapely.readthedocs.io) - A python library which provides functionality from the GEOS library for manipulating vector geometries.
* [Fiona](http://toblerity.org/fiona) - A python library for the manipulation of vector data.
* [pyroSAR](https://pyrosar.readthedocs.io) - The pyroSAR package aims at providing a complete solution for the scalable organization and processing of SAR satellite data.
* [PyRAT](https://github.com/birgander2/PyRAT) - General purpose Synthetic Aperture Radar (SAR) postprocessing software package

### Machine Learning Python Modules
* [scikit-learn](http://scikit-image.org) - A powerful and easy to use machine learning library which can be linked with RIOS to perform many classification and clustering operations. Used a lot within RSGISLib to provide classification functionality.
* [tensorflow](https://www.tensorflow.org) - A core open source library to help you develop and train ML models.
* [PyTorch](https://pytorch.org) - An open source machine learning framework that accelerates the path from research prototyping to production deployment.
* [PyOD](https://pyod.readthedocs.io) - PyOD is a comprehensive and scalable Python toolkit for detecting outlying objects in multivariate data.

### Models
* [DART](http://www.cesbio.ups-tlse.fr/us/dart.html) - DART models radiative transfer in the system "Earth - Atmosphere", from visible to thermal infrared. It simulates measurements (images, waveforms,…) of passive and active (lidar) satellite/plane sensors, as well as the radiative budget, for urban and natural landscapes.
* [Py6S](https://py6s.readthedocs.io) - Py6S is a interface to the Second Simulation of the Satellite Signal in the Solar Spectrum (6S) atmospheric Radiative Transfer Model through the Python programming language.
