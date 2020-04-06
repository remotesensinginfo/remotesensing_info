title: Introduction to RSGISLib Tutorial
category: Earth Observation
tags: Python, RSGISLib, Image Processing.
date: 2020-01-24
modified: 2020-01-24


------------------------------------------------------------------------

This work (including scripts) is licensed under a Creative Commons Attribution 4.0 International License. To view a copy of this license, visit <http://creativecommons.org/licenses/by/4.0/>. With the exception of PALSAR images which are copyright © JAXA and Landsat images which are copyright © USGS.

------------------------------------------------------------------------

Introduction
============

This course aims to provide an introduction to the Remote Sensing and GIS Library (RSGISLib; Bunting et al., 2014) developed by Pete Bunting (<pfb@aber.ac.uk>) and Dan Clewley (<dac@pml.ac.uk>), and the stack of open source software commonly used in combination with it.

Specifically, by the end of this course you should have knowledge of:

-   using band maths through RSGISLib

-   pre-processing remote sensing imagery including:

    -   mosaicking

    -   subsetting

    -   stacking image bands

    -   subsetting image bands

    -   resampling imagery

    -   creating a valid mask

-   converting to and from vector and raster data

-   filtering images

-   perform a random forests per-pixel classification using scikit-learn

-   perform an image segmentation

-   perform a random forest segment based classification using
    scikit-learn

In addition to RSGISLib, this worksheet makes use of the open source Geospatial Data Abstraction Library (GDAL; <http://www.gdal.org>), the TuiView image viewer (<http://www.tuiview.org>) the Raster Input/Output Simplification library (RIOS) library (<http://www.rioshome.org>) and scikit-learn (<http://scikit-learn.org>) machine learning library. These software are freely available and can be used on a wide variety of computing platforms from desktops to large scale high performance computing (HPC) systems and cloud systems such as Amazon or Google. Together they form a system for performing spatial image analysis, primarily derived from remote sensing imagery (Clewley et al., 2014).

1.  Geospatial Data Abstraction Library (GDAL; <http://www.gdal.org>)

2.  The Remote Sensing and GIS software library (RSGISLib;
    <http://www.rsgislib.org>)

3.  KEA file format (<http://www.kealib.org>)

4.  TuiView (<http://www.tuiview.org>)

5.  Python (<http://www.python.org>)

6.  RIOS (<http://www.rioshome.org>)

7.  scikit-learn (<http://scikit-learn.org>)

Getting Help / Documentation
============================

Online Documentation
--------------------

Online documentation for the latest version of RSGISLib is provided on the website (<http://www.rsgislib.org>) and should be your first point of call. When using RSGISLib you'll find that this website provides a crucial reference for the available functions.

There are also other tutorials on the website under (http://www.rsgislib.org/documentation.html).

Blog
----

You will also have the two online blogs:

<https://spectraldifferences.wordpress.com> a useful source of examples of specific problems (e.g., processing PALSAR data https://spectraldifferences.wordpress.com/tag/palsar/).

<https://www.remotesensing.info> providing guides on how to install and setup software (<https://www.remotesensing.info/pages/Software.html> and links to further training materials.

Mailing List
------------

We have a mailing list
<https://groups.google.com/forum/#!forum/rsgislib-support> where you can communicate with others using RSGISLib and associated tools if you have specific questions or think there is bug or problem with the software. We do our best to answer emails on this list in a prompt manner. To post ([rsgislib-support@googlegroups.com](rsgislib-support@googlegroups.com)) on this mailing list you first need to register.

Code Repository
---------------

If you would like to see the RSGISLib source code or submit to our issues list this is done through the bitbucket service, (https://bitbucket.org/petebunting/rsgislib).

Getting Started
===============

To get started we will have a look at the files provided within the `Scripts` and `Datasets` directories. The `Scripts` directory has all the scripts used within the course, with the order they are referred to in this document specified with the preceding number. The datasets directory includes all the data required for this course (Table 1).

| Dataset                         | Description                                                  |
| ------------------------------- | ------------------------------------------------------------ |
| ALOS PALSAR-2                   | JAXA Mosaic data from 2015.                                  |
| Landsat-8                       | USGS Landsat-8 imagery, two scenes which have been atmospherically corrected to surface reflectance and merged. |
| Training data (ESRI Shapefiles) | These were manually drawn with reference to Google Earth imagery using QGIS |

To run RSGISLib we need to create a python script (a text file with the file extension `.py`) in which you just list the RSGISLib functions you wish to run. You can think of an RSGISLib function like a dialog box for running a command in a GUI based application (e.g., ENVI / Erdas Imagine / ArcGIS) where you provide all the options required (e.g., input and output files). However, as you can list a series of commands which all  run one after another you can streamline your data processing chains ensuring that all files are run with the same options and by looping through a number of input files easily batch process your data. We will first create a working directory on your system, for example on your Desktop, `RSGISLibTraining` where we will do our work:

``` bash
cd Desktop
mkdir RSGISLibTraining
cd RSGISLibTraining
```

Image Band Maths
----------------

In this example we will calculate a three band composite for a PALSAR-2 scene. The output image will have three image bands:

-   HH Power

-   HV Power

-   HH/HV

Before we start, create a directory for this exercise your working space and copy the file `N22E089_15_MOS_F02DAR.tar.gz` into that directory.

``` bash
cd Desktop/RSGISLibTraining
mkdir Exercise1
cd Exercise1
```

Extract `N22E089_15_MOS_F02DAR.tar.gz` using the `tar` command:

``` python
tar -zxf ./N22E089_15_MOS_F02DAR.tar.gz
```

To completed the processing steps I find it helpful to first write out the steps in coding comments, breaking the problem down. The steps for this problem are the following:

``` python
#/usr/bin/env python

# Import python modules

# Calculate Power for HH image

# Calculate Power for HV image

# Calculate HH / HV

# Stack images into a single band

# Calculate overview pyramids and image statistics to make visualisation faster.
```

Producing the solution to those is now a case of 'filling out the form'. In this case, those steps are filled using the following rsgislib functions:

``` python
#/usr/bin/env python

# Import python modules
import rsgislib
from rsgislib import imagecalc
from rsgislib import imageutils

# Calculate Power for HH image
hhImg = 'N22E089_15_sl_HH_F02DAR'
hhImgPow = 'N22E089_15_sl_HH_F02DAR_pow.kea'
bandDefns = []
bandDefns.append(imagecalc.BandDefn('HH', hhImg, 1))
mathExp = 'HH>0?10^(2*log10(HH) - 8.3):0.0'
imagecalc.bandMath(hhImgPow, mathExp, 'KEA', rsgislib.TYPE_32FLOAT, bandDefns)

# Calculate Power for HV image
hvImg = 'N22E089_15_sl_HV_F02DAR'
hvImgPow = 'N22E089_15_sl_HV_F02DAR_pow.kea'
bandDefns = []
bandDefns.append(imagecalc.BandDefn('HV', hvImg, 1))
mathExp = 'HV>0?10^(2*log10(HV) - 8.3):0.0'
imagecalc.bandMath(hvImgPow, mathExp, 'KEA', rsgislib.TYPE_32FLOAT, bandDefns)

# Calculate HH / HV
hhhvImg = 'N22E089_15_sl_HHHV_F02DAR_pow.kea'
bandDefns = []
bandDefns.append(imagecalc.BandDefn('HH', hhImgPow, 1))
bandDefns.append(imagecalc.BandDefn('HV', hvImgPow, 1))
mathExp = 'HV>0?HH/HV:0.0'
imagecalc.bandMath(hhhvImg, mathExp, 'KEA', rsgislib.TYPE_32FLOAT, bandDefns)

# Stack images into a single band
imageList = [hhImgPow, hvImgPow, hhhvImg]
bandNamesList = ['HH','HV', 'HH/HV']
outputImageStack = 'N22E089_15_sl_F02DAR_powstack.kea'
imageutils.stackImageBands(imageList, bandNamesList, outputImageStack, None, 0, 
                           'KEA', rsgislib.TYPE_32FLOAT)

# Calculate overview pyramids and image statistics to make visualisation faster.
imageutils.popImageStats(outputImageStack, usenodataval=True, nodataval=0, 
                         calcpyramids=True)
```

The script file `01_BandMaths.py` has this code, copy it into your working directory `Exercise1` and run it.

``` bash
python 01_BandMaths.py
```

If helpful, you can watch these steps being executed using the following video:

- https://www.youtube.com/watch?v=-c5mLH5rIws

View the end result using the Tuiview image viewer.  You start Tuiview use the Terminal:

``` bash
tuiview
```

You can provide the filename of the file you wish to open on the terminal as well if you wish (this makes it particularly useful for viewing imagery on a remote server over '`ssh -X`').

``` bash
tuiview N22E089_15_sl_F02DAR_powstack.kea
```

If you have not used Tuiview before or are unsure how to use it, have a
look a these online videos:

-   Introduction to Tuiview:
    <https://www.youtube.com/watch?v=2VXH-hxDfvg>

-   Stretching Multiple Images:
    <https://www.youtube.com/watch?v=McZdFa49q7o>

-   Using Tuiview from Terminal:
    <https://www.youtube.com/watch?v=kbVu4JeEsdU>

-   Using Multiple Windows:
    <https://www.youtube.com/watch?v=Rau0LH-YLUU>

### Tidy up your files

You can now delete the directory `Exercise1` to save disk space.

``` bash
rm -Rf Exercise1
```

Batch Processing
----------------

If you want to apply this process to a large number of image datasets
we'd need to add some steps to our processing chain:

``` python
#/usr/bin/env python

# Import python modules

# Extract tar.gz file

# Find the HH and HV images.

# Calculate Power for HH image

# Calculate Power for HV image

# Calculate HH / HV

# Stack images into a single band

# Calculate overview pyramids and image statistics to make visualisation faster.

# Clean up so only the stack remains.
```

To do this it is useful to define a function which does the work which
can be called multiple times, once for each input file:

``` python
#/usr/bin/env python

# Import python modules

def createPALSARStack(inputTAR, outputStackImg, tmpDir):
    # Extract tar.gz file
    
    # Find the HH and HV images.
    
    # Calculate Power for HH image
    
    # Calculate Power for HV image
    
    # Calculate HH / HV
    
    # Stack images into a single band
    
    # Calculate overview pyramids and image statistics to make visualisation faster.
    
    # Clean up so only the stack remains.

createPALSARStack('N22E088_15_MOS_F02DAR.tar.gz', 'N22E088_15_MOS_F02DAR_Stack.kea', ./tmp')
createPALSARStack('N22E089_15_MOS_F02DAR.tar.gz', 'N22E089_15_MOS_F02DAR_Stack.kea', ./tmp')
createPALSARStack('N23E088_15_MOS_F02DAR.tar.gz', 'N23E088_15_MOS_F02DAR_Stack.kea', ./tmp')
createPALSARStack('N23E089_15_MOS_F02DAR.tar.gz', 'N23E089_15_MOS_F02DAR_Stack.kea', ./tmp')
```

`02A_PALSARStack.py` (below) provides the code which which will create a
temporary working directory, extract the TAR file and find the HH and HV
files before deleting the temporary working directory to clean up the
file.

``` python
#/usr/bin/env python

# Import python modules
import rsgislib
from rsgislib import imagecalc
from rsgislib import imageutils

import os
import os.path
import subprocess
import shutil
import glob

def createPALSARStack(inputTAR, outputStackImg, tmpDir):
    
    # Make sure that the inputs use absolute paths as using a working directory
    inputTAR = os.path.abspath(inputTAR)
    outputStackImg = os.path.abspath(outputStackImg)
    tmpDir = os.path.abspath(tmpDir)
    
    # Check file input file and tmp directory are present.
    if not os.path.exists(tmpDir):
        raise rsgislib.RSGISPyException('Tmp directory is not present.')
        
    if not os.path.exists(inputTAR):
        raise rsgislib.RSGISPyException('Input tar file is not present.')
    # Get the rsgis utilities object
    rsgisUtils = rsgislib.RSGISPyUtils()
    
    # Get a unique id for processing
    uidStr = rsgisUtils.uidGenerator()
    
    # Create a working directory
    workDIR = os.path.join(tmpDir, uidStr)
    if not os.path.exists(workDIR):
        os.makedirs(workDIR)
    # Save current working path
    cPWD = os.getcwd()
    # Move into that working directory.
    os.chdir(workDIR)
    
    # Extract tar.gz file - using the terminal commands.
    cmd = 'tar -zxf {}'.format(inputTAR)
    print(cmd)
    try:
        subprocess.call(cmd, shell=True)
    except OSError as e:
       raise rsgislib.RSGISPyException('Could not execute command: ' + cmd)
    
    # Find the HH and HV images.
    hhImg = ''
    hvImg = ''
    
    hhFiles = glob.glob(os.path.join(workDIR, '*_sl_HH_F02DAR'))
    hvFiles = glob.glob(os.path.join(workDIR, '*_sl_HV_F02DAR'))
    
    if len(hhFiles) == 1:
        hhImg = hhFiles[0]
    else:
        raise rsgislib.RSGISPyException('Could not find HH file')
    if len(hvFiles) == 1:
        hvImg = hvFiles[0]
    else:
        raise rsgislib.RSGISPyException('Could not find HV file')
    
    print("HH File: ", hhImg)
    print("HV File: ", hvImg)
    
    # Calculate Power for HH image
    
    # Calculate Power for HV image
    
    # Calculate HH / HV
    
    # Stack images into a single band
    
    # Calculate overview pyramids and image statistics to make visualisation faster.
                  
    # Clean up by deleting the working directory
    os.chdir(cPWD) # Move back to starting directory before delete
    shutil.rmtree(workDIR)
    
createPALSARStack('N22E088_15_MOS_F02DAR.tar.gz', 'N22E088_15_MOS_F02DAR_Stack.kea', './tmp')
createPALSARStack('N22E089_15_MOS_F02DAR.tar.gz', 'N22E089_15_MOS_F02DAR_Stack.kea', './tmp')
createPALSARStack('N23E088_15_MOS_F02DAR.tar.gz', 'N23E088_15_MOS_F02DAR_Stack.kea', './tmp')
createPALSARStack('N23E089_15_MOS_F02DAR.tar.gz', 'N23E089_15_MOS_F02DAR_Stack.kea', './tmp')

```

Copy the all the tar.gz PALSAR-2 files and the `02A_PALSARStack.py` into
your working directory under a directory `Exercise2` and then run the
`02A_PALSARStack.py` script.

``` bash
python 02A_PALSARStack.py
```

You should get an output similar to that below:

``` bash
(osgeo) WS0978:Exercise2 pete$ python 02A_PALSARStack.py 
tar -zxf /Users/pete/Desktop/RSGISLibTraining/Exercise2/N22E088_15_MOS_F02DAR.tar.gz
HH File:  /Users/pete/Desktop/RSGISLibTraining/Exercise2/tmp/e81517/N22E088_15_sl_HH_F02DAR
HV File:  /Users/pete/Desktop/RSGISLibTraining/Exercise2/tmp/e81517/N22E088_15_sl_HV_F02DAR
tar -zxf /Users/pete/Desktop/RSGISLibTraining/Exercise2/N22E089_15_MOS_F02DAR.tar.gz
HH File:  /Users/pete/Desktop/RSGISLibTraining/Exercise2/tmp/8e9c34/N22E089_15_sl_HH_F02DAR
HV File:  /Users/pete/Desktop/RSGISLibTraining/Exercise2/tmp/8e9c34/N22E089_15_sl_HV_F02DAR
tar -zxf /Users/pete/Desktop/RSGISLibTraining/Exercise2/N23E088_15_MOS_F02DAR.tar.gz
HH File:  /Users/pete/Desktop/RSGISLibTraining/Exercise2/tmp/5b4716/N23E088_15_sl_HH_F02DAR
HV File:  /Users/pete/Desktop/RSGISLibTraining/Exercise2/tmp/5b4716/N23E088_15_sl_HV_F02DAR
tar -zxf /Users/pete/Desktop/RSGISLibTraining/Exercise2/N23E089_15_MOS_F02DAR.tar.gz
HH File:  /Users/pete/Desktop/RSGISLibTraining/Exercise2/tmp/01736a/N23E089_15_sl_HH_F02DAR
HV File:  /Users/pete/Desktop/RSGISLibTraining/Exercise2/tmp/01736a/N23E089_15_sl_HV_F02DAR
```

We can now combine the two scripts `01_BandMaths.py` and
`02A_PALSARStack.py` to run the whole process creating
`02B_PALSARStack.py`:

``` python
#/usr/bin/env python

# Import python modules
import rsgislib
from rsgislib import imagecalc
from rsgislib import imageutils

import os
import os.path
import subprocess
import shutil
import glob

def createPALSARStack(inputTAR, outputStackImg, tmpDir):
    
    # Make sure that the inputs use absolute paths as using a working directory
    inputTAR = os.path.abspath(inputTAR)
    outputStackImg = os.path.abspath(outputStackImg)
    tmpDir = os.path.abspath(tmpDir)
    
    # Check file input file and tmp directory are present.
    if not os.path.exists(tmpDir):
        raise rsgislib.RSGISPyException('Tmp directory is not present.')
        
    if not os.path.exists(inputTAR):
        raise rsgislib.RSGISPyException('Input tar file is not present.')
    # Get the rsgis utilities object
    rsgisUtils = rsgislib.RSGISPyUtils()
    
    # Get a unique id for processing
    uidStr = rsgisUtils.uidGenerator()
    
    # Create a working directory
    workDIR = os.path.join(tmpDir, uidStr)
    if not os.path.exists(workDIR):
        os.makedirs(workDIR)
    # Save current working path
    cPWD = os.getcwd()
    # Move into that working directory.
    os.chdir(workDIR)
    
    # Extract tar.gz file - using the terminal commands.
    cmd = 'tar -zxf {}'.format(inputTAR)
    print(cmd)
    try:
        subprocess.call(cmd, shell=True)
    except OSError as e:
       raise rsgislib.RSGISPyException('Could not execute command: ' + cmd)
    
    # Find the HH and HV images.
    hhImg = ''
    hvImg = ''
    
    hhFiles = glob.glob(os.path.join(workDIR, '*_sl_HH_F02DAR'))
    hvFiles = glob.glob(os.path.join(workDIR, '*_sl_HV_F02DAR'))
    
    if len(hhFiles) == 1:
        hhImg = hhFiles[0]
    else:
        raise rsgislib.RSGISPyException('Could not find HH file')
    if len(hvFiles) == 1:
        hvImg = hvFiles[0]
    else:
        raise rsgislib.RSGISPyException('Could not find HV file')
    
    print("HH File: ", hhImg)
    print("HV File: ", hvImg)
    
    # Calculate Power for HH image
    hhImgPow = os.path.join(workDIR, uidStr+'_HH_Pow.kea')
    bandDefns = []
    bandDefns.append(imagecalc.BandDefn('HH', hhImg, 1))
    mathExp = 'HH>0?10^(2*log10(HH) - 8.3):0.0'
    imagecalc.bandMath(hhImgPow, mathExp, 'KEA', rsgislib.TYPE_32FLOAT, bandDefns)
    
    # Calculate Power for HV image
    hvImgPow = os.path.join(workDIR, uidStr+'_HV_Pow.kea')
    bandDefns = []
    bandDefns.append(imagecalc.BandDefn('HV', hvImg, 1))
    mathExp = 'HV>0?10^(2*log10(HV) - 8.3):0.0'
    imagecalc.bandMath(hvImgPow, mathExp, 'KEA', rsgislib.TYPE_32FLOAT, bandDefns)

    # Calculate HH / HV
    hhhvImg = os.path.join(workDIR, uidStr+'_HHHV_Pow.kea')
    bandDefns = []
    bandDefns.append(imagecalc.BandDefn('HH', hhImgPow, 1))
    bandDefns.append(imagecalc.BandDefn('HV', hvImgPow, 1))
    mathExp = 'HV>0?HH/HV:0.0'
    imagecalc.bandMath(hhhvImg, mathExp, 'KEA', rsgislib.TYPE_32FLOAT, bandDefns)

    # Stack images into a single band
    imageList = [hhImgPow, hvImgPow, hhhvImg]
    bandNamesList = ['HH','HV', 'HH/HV']
    imageutils.stackImageBands(imageList, bandNamesList, outputStackImg, None, 0, 
                               'KEA', rsgislib.TYPE_32FLOAT)
    
    # Calculate overview pyramids and image statistics to make visualisation faster.
    imageutils.popImageStats(outputStackImg, usenodataval=True, nodataval=0, 
                         calcpyramids=True)
                  
    # Clean up by deleting the working directory
    os.chdir(cPWD) # Move back to starting directory before delete
    shutil.rmtree(workDIR)
    
createPALSARStack('N22E088_15_MOS_F02DAR.tar.gz', 'N22E088_15_MOS_F02DAR_Stack.kea', './tmp')
createPALSARStack('N22E089_15_MOS_F02DAR.tar.gz', 'N22E089_15_MOS_F02DAR_Stack.kea', './tmp')
createPALSARStack('N23E088_15_MOS_F02DAR.tar.gz', 'N23E088_15_MOS_F02DAR_Stack.kea', './tmp')
createPALSARStack('N23E089_15_MOS_F02DAR.tar.gz', 'N23E089_15_MOS_F02DAR_Stack.kea', './tmp')

```

Copy `02B_PALSARStack.py` into your `Exercise2` working directory and
run it.

``` bash
python 02B_PALSARStack.py
```

If helpful, you can watch these steps being executed in the following video:

- https://www.youtube.com/watch?v=VRD_douyvww

You can now load you images into tuiview

``` bash
tuiview *.kea
```

Tidy up your files
------------------

To tidy up your files and save space only the final stacked KEA files from your `Exercise2` directory are needed for the next exercises, the rest can be deleted.

Data Pre-processing
===================

Stacking and converting the SAR data to power is part of the pre-processing ahead of doing something 'useful' with the image data. However, there are many other processes which can be required or helpful
in preparing data for the application, in this case classification.

To start the analysis create a new directory (`Exercise3`) in your working space and copy the KEA image files outputted from the previous step into that directory.

``` bash
cd Desktop/RSGISLibTraining
mkdir Exercise3
cd Exercise3
cp ../Exercise2/*.kea .
```

For this next section, the following video has been created showing the processing steps being executed:

- https://www.youtube.com/watch?v=dkamKMSGcnU

Image Mosaicking
----------------

The first process is to mosaic the four PALSAR-2 images to create a single image file, the code shown below.

``` python
#/usr/bin/env python

# Import python modules
import rsgislib
from rsgislib import imageutils

# List of the files to be mosaicked.
inputList = ['N22E088_15_MOS_F02DAR_Stack.kea', 
             'N22E089_15_MOS_F02DAR_Stack.kea', 
             'N23E088_15_MOS_F02DAR_Stack.kea', 
             'N23E089_15_MOS_F02DAR_Stack.kea']
outImage = './Sundarbans_15_MOS_F02DAR.kea'
imageutils.createImageMosaic(inputList, outImage, 0.0, 0.0, 1, 0, 'KEA', rsgislib.TYPE_32FLOAT)
imageutils.popImageStats(outImage, usenodataval=True, nodataval=0, calcpyramids=True)

```

See the documentation, for explanation of the different inputs:

- <http://www.rsgislib.org/rsgislib_imageutils.html#rsgislib.imageutils.createImageMosaic>

Please note, you can also use the `glob` module to automatically find the input files. For example, see script `03B_MosaicImages.py` , remember to delete the previous output (`Sundarbans_15_MOS_F02DAR.kea`)
before running this script.

``` python
#/usr/bin/env python

# Import python modules
import rsgislib
from rsgislib import imageutils

import glob

# Search for all files with the extension 'kea'
inputList = glob.glob('*.kea')
outImage = './Sundarbans_15_MOS_F02DAR.kea'
imageutils.createImageMosaic(inputList, outImage, 0.0, 0.0, 1, 0, 'KEA', rsgislib.TYPE_32FLOAT)
imageutils.popImageStats(outImage, usenodataval=True, nodataval=0, calcpyramids=True)

```

Re-projecting / Re-sampling images
----------------------------------

In the classification we are going to use PALSAR-2 and Landsat-8 imagery. However, the PALSAR-2 imagery is provided in lat/long WGS84 while the Landsat-8 imagery is provided in WSG84 UTM Zone 45. We are
therefore going to re-project and resample the PALSAR-2 imagery to match the Landsat-8 imagery, this will also mean our coordinates will be in metres, which is more convenient.

``` python
#/usr/bin/env python

# Import python modules
import rsgislib
from rsgislib import imageutils

inRefImg = 'Sundarbans_201511_Landsat.kea'
inProcessImg = 'Sundarbans_15_MOS_F02DAR.kea'
outImg = 'Sundarbans_15_MOS_F02DAR_utm45n.kea'

imageutils.resampleImage2Match(inRefImg, inProcessImg, outImg, 'KEA', 'cubic',
                               rsgislib.TYPE_32FLOAT)
imageutils.popImageStats(outImg, usenodataval=True, nodataval=0, calcpyramids=True)

```

The documentation for this command is available here:

- http://www.rsgislib.org/rsgislib_imageutils.html#rsgislib.imageutils.resampleImage2Match

The command can be used where ever you need one image to match another in terms of the pixel grid, pixel size, extent and projection. If you want to re-project without providing an image file to match to then you can use `rsgislib.imageutils.reprojectImage`. Other useful commands for dealing with projection information include:

-   `rsgislib.imageutils.assignProj` -- assigned a wkt file to be the
    projection representation for the input image.

-   `rsgislib.imageutils.assignSpatialInfo` -- assign the spatial header
    information for the input image.

-   `rsgislib.imageutils.copyProjFromImage` -- copy the projection
    representation from one image and assign to another.

-   `rsgislib.imageutils.copySpatialAndProjFromImage` -- copy the
    projection and spatial header from one image and assigned to
    another.

Create Valid Data Extent
------------------------

The following script creates a valid data mask for both the PALSAR-2 and Landsat-8 imagery. The valid data command outputs a binary mask, where 1 means that the pixel is valid data and 0 not. Valid data is defined as a pixel where no band is assigned the no data value (i.e., in this case 0).

``` python
#/usr/bin/env python

# Import python modules
import rsgislib
from rsgislib import imageutils
from rsgislib import rastergis

landsatImg = 'Sundarbans_201511_Landsat.kea'
palsarImg = 'Sundarbans_15_MOS_F02DAR_utm45n.kea'
validMask = 'Sundarbans_validmsk.kea'

imageutils.genValidMask(inimages=[landsatImg,palsarImg], outimage=validMask, gdalformat='KEA', nodata=0.0)

rastergis.populateStats(clumps=validMask, addclrtab=True, calcpyramids=True, ignorezero=True) 

```

Raster-to-Vector
----------------

We will now use the raster to vector function to create a polygon shapefile for the area of valid data:

``` python
#/usr/bin/env python

# Import python modules
import rsgislib
from rsgislib import vectorutils

inputImg = 'Sundarbans_validmsk.kea'
outVecFile = 'Sundarbans_validmsk_vec.shp'  
outVecLyr = 'Sundarbans_validmsk_vec'              

vectorutils.polygoniseRaster2VecLyr(outVecFile, outVecLyr, 'ESRI Shapefile', inputImg, 
                                    imgBandNo=1, maskImg=inputImg, imgMaskBandNo=1)

```

The documentation is here:

- http://www.rsgislib.org/rsgislib_vectorutils.html#rsgislib.vectorutils.polygoniseRaster

### Vector-to-Raster

Within RSGISLib we commonly use rasters in preference of vectors and therefore rasterising a vector on to same pixel grid as the imagery we intend to use the vector layer with is a common operation. In this case,
we demonstrate rasterising the vector we just extracted from the previous step but on to the image grid of the Landsat image (i.e., rather than the combined valid mask image which is a different (common) extent).

``` python
#/usr/bin/env python

# Import python modules
import rsgislib
from rsgislib import vectorutils

inVecFile = 'Sundarbans_validmsk_vec.shp'  
inVecLyr = 'Sundarbans_validmsk_vec'  
inputImage = 'Sundarbans_201511_Landsat.kea'
outImage = 'Sundarbans_ValidMask_Landsat_tmp.kea'
vectorutils.rasteriseVecLyr(inVecFile, inVecLyr, inputImage, outImage,  
                            gdalformat="KEA", burnVal=1)
    
```

The documentation for this command is available here
<http://www.rsgislib.org/rsgislib_vectorutils.html#rsgislib.vectorutils.polygoniseRaster>.
Please note that this function is using the gdal\_rasterize command
(<http://www.gdal.org/gdal_rasterize.html>) and therefore the
documentation for that might also be a useful reference.

The following commands are also worth being aware of:

-   `rsgislib.rastergis.importVecAtts` -- allows attribute information
    from the vector to be copied to a raster file (i.e., raster
    attribute table). You must be using KEA or HFA file types for this
    command.

-   `rsgislib.vectorutils.copyShapefile2RAT` -- This command combines
    the two command to rasterize a vector file and copy all the
    attributes to the output raster.

Image Spatial Subsetting
------------------------

The next command allows you to subset to a region of interest an image file, in this case copy the shapefile generated from the previous step. We could have also used the valid image mask (`Sundarbans_validmsk.kea`) directly using the `rsgislib.imageutils.subset2img` function, but for the sake of the course I wanted to illustrate the vector-to-raster and raster-to-vector functions. You can now run the script`07_Subset2ROI.py`:

``` python
#/usr/bin/env python

# Import python modules
import rsgislib
from rsgislib import imagecalc
from rsgislib import imageutils

inputimage = 'Sundarbans_201511_Landsat_sub_msk.kea'

outputimage = 'Sundarbans_201511_Landsat_sub_msk_ndvi.kea'
bandDefns = [imagecalc.BandDefn('red', inputimage, 4),
             imagecalc.BandDefn('nir', inputimage, 5)]
imagecalc.bandMath(outputimage, 'nir==0?999:(nir-red)/(nir+red)', 'KEA', 
                   rsgislib.TYPE_32FLOAT, bandDefns)
imageutils.popImageStats(outputimage, usenodataval=True, nodataval=999, calcpyramids=True)
                         
outputimage = 'Sundarbans_201511_Landsat_sub_msk_wbi.kea'                    
bandDefns = [imagecalc.BandDefn('blue', inputimage, 2),
             imagecalc.BandDefn('nir', inputimage, 5)]
imagecalc.bandMath(outputimage, 'nir==0?999:(blue/nir)', 'KEA', 
                   rsgislib.TYPE_32FLOAT, bandDefns)
imageutils.popImageStats(outputimage, usenodataval=True, nodataval=999, calcpyramids=True)

```

The documentation for this command is available here:

- http://www.rsgislib.org/rsgislib_imageutils.html#rsgislib.imageutils.subset

This command subsets to the extend on a shapefile, however there are also commands to subset using other datasets:

-   `rsgislib.imageutils.subset2img` -- subsets an image to the same extent as another image.
    
-   `rsgislib.imageutils.subsetbbox` -- subsets an image to the extent defined by the bounding box.
    
-   `rsgislib.imageutils.subset2polys` -- creates multiple output images, matching the number of polygons in the input file. Each output image is subsetted to the extent of one of the polygons in
    the input shapefile.
    
-   `rsgislib.imageutils.subsetImgs2CommonExtent` -- calculates the union in extent of all the images and subsets them all to that extent.
    -   `rsgislib.imageutils.buildImgSubDict` -- can be used to automatically build the input for the
        `rsgislib.imageutils.subsetImgs2CommonExtent` command.

Image Masking
-------------

Masking an image is a commonly applied function to eliminate parts of the image data we do not want to consider. RSGISLib has a flexible masking command:

- <http://www.rsgislib.org/rsgislib_imageutils.html#rsgislib.imageutils.maskImage>

This is used below to mask both the landsat and palsar-2 data to the valid region of both.

``` python
#/usr/bin/env python

# Import python modules
import rsgislib
from rsgislib import imageutils

imagemask = 'Sundarbans_validmsk.kea'

inputimage = 'Sundarbans_15_MOS_F02DAR_utm45n_sub.kea'
outputimage = 'Sundarbans_15_MOS_F02DAR_utm45n_sub_msk.kea'

imageutils.maskImage(inputimage, imagemask, outputimage, 'KEA', 
                     rsgislib.TYPE_32FLOAT, 0.0, 0.0)
imageutils.popImageStats(outputimage, usenodataval=True, nodataval=0, calcpyramids=True)
                         
inputimage = 'Sundarbans_201511_Landsat_sub.kea'
outputimage = 'Sundarbans_201511_Landsat_sub_msk.kea'

imageutils.maskImage(inputimage, imagemask, outputimage, 'KEA', 
                     rsgislib.TYPE_16UINT, 0.0, 0.0)
imageutils.popImageStats(outputimage, usenodataval=True, nodataval=0, calcpyramids=True)

```

Please note when you are masking, the final two input parameters `outvalue` the value which is going to be given to the pixels which are being 'masked out' during the process and `maskvalue` which is the pixel value(s) in the mask file which correspond to the regions in the input image which will be 'masked out' and assigned the `outvalue`. It is worth noting that `maskvalue` can also be a list of values (e.g., `[1,2,3]`), when a list of value is provided all the values are treat the same and replaced with the same `outvalue`.

Image Band Subsetting
---------------------

RSGISLib can also subset in terms of the image bands in the file, in this case for converting to dBs we just want the HH and HV bands (i.e., 1 and 2). The following script performs that operation.

``` python
#/usr/bin/env python

# Import python modules
import rsgislib
from rsgislib import imageutils

inputimage = 'Sundarbans_15_MOS_F02DAR_utm45n_sub_msk.kea'
outputimage = 'Sundarbans_15_MOS_F02DAR_utm45n_sub_msk_HHHV.kea'
imageutils.selectImageBands(inputimage, outputimage, 'KEA', rsgislib.TYPE_32FLOAT, [1,2])
imageutils.popImageStats(outputimage, usenodataval=True, nodataval=0, calcpyramids=True)

```

Apply Lee Filter
----------------

For many applications if can be useful to filter the SAR imagery, RSGISLib has a number of filters available
(<http://www.rsgislib.org/rsgislib_imagefilter.html>) but in this case a Lee filter is applied.

``` python
#/usr/bin/env python

# Import python modules
import rsgislib
from rsgislib import imageutils
from rsgislib import imagefilter

inputimage = 'Sundarbans_15_MOS_F02DAR_utm45n_sub_msk_HHHV.kea'
outputimage = 'Sundarbans_15_MOS_F02DAR_utm45n_sub_msk_HHHV_lee.kea'
imagefilter.applyLeeFilter(inputimage, outputimage, 5, 5, 'KEA', rsgislib.TYPE_32FLOAT)
imageutils.popImageStats(outputimage, usenodataval=True, nodataval=0, calcpyramids=True)

```

If you have a very large image, filtering can be very slow. Therefore, RSGISLib provides a tiled base filtering option (`rsgislib.imagefilter.tiledfilter.performTiledImgMultiFilter`) which first tiles the input image, with an appropriate overlap, filters each of the tiles, using multiple processing cores if requested, and then mosaics the results ignoring the overlap resulting in a seamless result but much faster for large images.

Band Maths: Power -- dBs
------------------------

Before classification we also want to convert the HH and HV bands to dBs, in this case we'll use the `rsgislib.imagecalc.imageMath` function rather than `rsgislib.imagecalc.bandMath` that was shown earlier as it outputs a multi-band image where the same expression is applied to all the bands. In the expression the band must be referred to as `b1`. The following script converts both the filtered and unfiltered image to dBs.

``` python
#/usr/bin/env python

# Import python modules
import rsgislib
from rsgislib import imageutils
from rsgislib import imagecalc

inputimage = 'Sundarbans_15_MOS_F02DAR_utm45n_sub_msk_HHHV.kea'
outputimage = 'Sundarbans_15_MOS_F02DAR_utm45n_sub_msk_HHHV_dB.kea'
imagecalc.imageMath(inputimage, outputimage, 'b1>0?10*log10(b1):999', 'KEA', 
                    rsgislib.TYPE_32FLOAT)
imageutils.popImageStats(outputimage, usenodataval=True, nodataval=999, calcpyramids=True)
                         
                         
inputimage = 'Sundarbans_15_MOS_F02DAR_utm45n_sub_msk_HHHV_lee.kea'
outputimage = 'Sundarbans_15_MOS_F02DAR_utm45n_sub_msk_HHHV_lee_dB.kea'
imagecalc.imageMath(inputimage, outputimage, 'b1>0?10*log10(b1):999', 'KEA', 
                    rsgislib.TYPE_32FLOAT)
imageutils.popImageStats(outputimage, usenodataval=True, nodataval=999, calcpyramids=True)

```

Band Maths: NDVI & WBI
----------------------

From optical image the normalised difference vegetation index (NDVI) and water band index (WBI) are useful indices which are commonly applied. These can be calculate as shown below using the `imagecalc.bandmaths`function.

``` python
#/usr/bin/env python

# Import python modules
import rsgislib
from rsgislib import imagecalc
from rsgislib import imageutils

inputimage = 'Sundarbans_201511_Landsat_sub_msk.kea'

outputimage = 'Sundarbans_201511_Landsat_sub_msk_ndvi.kea'
bandDefns = [imagecalc.BandDefn('red', inputimage, 4),
             imagecalc.BandDefn('nir', inputimage, 5)]
imagecalc.bandMath(outputimage, 'nir==0?999:(nir-red)/(nir+red)', 'KEA', 
                   rsgislib.TYPE_32FLOAT, bandDefns)
imageutils.popImageStats(outputimage, usenodataval=True, nodataval=999, calcpyramids=True)
                         
outputimage = 'Sundarbans_201511_Landsat_sub_msk_wbi.kea'                    
bandDefns = [imagecalc.BandDefn('blue', inputimage, 2),
             imagecalc.BandDefn('nir', inputimage, 5)]
imagecalc.bandMath(outputimage, 'nir==0?999:(blue/nir)', 'KEA', 
                   rsgislib.TYPE_32FLOAT, bandDefns)
imageutils.popImageStats(outputimage, usenodataval=True, nodataval=999, calcpyramids=True)

```

Putting it all together
-----------------------

It is worth stating that all those scripts could be combined to create a single script rather than multiple separate scripts. For easy of learning/understanding I have presented this course as individual scripts but if I was applying this analysis for myself I would have put together a single script. An overall script is provided in`13_CombinedPreProcessing.py` for illustration.

Tidy up your files
------------------

Following all that analysis you will have a number of intermediate files taking up disk space, clean that up so the only files you have remaining are:

-   `Sundarbans_15_MOS_F02DAR_utm45n_sub_msk.kea` -- HH, HV and HH/HV power image.
    
-   `Sundarbans_15_MOS_F02DAR_utm45n_sub_msk_HHHV.kea` -- HH and HV power image.
    
-   `Sundarbans_15_MOS_F02DAR_utm45n_sub_msk_HHHV_lee.kea` -- Lee filtered HH and HV power image.
    
-   `Sundarbans_15_MOS_F02DAR_utm45n_sub_msk_HHHV_dB.kea` -- HH and HV dB image.
    
-   `Sundarbans_15_MOS_F02DAR_utm45n_sub_msk_HHHV_lee_dB.kea` -- Lee filtered HH and HV dB image.
    
-   `Sundarbans_201511_Landsat_sub_msk.kea` -- Landsat image subsetted and masked.
    
-   `Sundarbans_201511_Landsat_sub_msk_wbi.kea` -- WBI Image.

-   `Sundarbans_201511_Landsat_sub_msk_ndvi.kea` -- NDVI image.

-   `Sundarbans_validmsk.kea` -- Valid pixel mask.

Image Classification
====================

Image classification is the process of assigning regions (either pixels or segments) to a thematic class based on the available information (i.e., backscatter, spectral reflectance etc.).

Classifications can be performed using a number of methods, on which there is a large body of literature, and the choice of approach is based on a number of factors. The primary factor is which one produces the
best (i.e., most accurate) result. However, in choosing a particular algorithm or approach the assumptions of that approach needs to be met. For example, when using Gaussian Maximum Likelihood you are making the assumption that your training data is normally distributed -- or at least close to being. Some algorithms require all the input data to be in the same value range, while others will be able to use categorical data within the classification and others won't. It is not in the scope of this document to consider those issues but please ensure that if you are not sure that you look up the classifier assumptions applying in
your own applications.

In terms of approaches you can group them as follows:

-   Rule Base -- i.e., manually defined decision trees.

-   Unsupervised -- i.e., clustering algorithms such as KMeans and
    ISOData.

-   Statistical Supervised

    -   Minimum Distance

    -   Paralleled Pipe

    -   Mahalanobis Distance

    -   Gaussian Maximum Likelihood

-   Machine Learning -- basically more advanced statistical supervised
    classifiers.

    -   K- nearest neighbour

    -   Decision Trees

    -   Support Vector Machines

    -   Random Forests

    -   Neural Networks

-   Deep Learning -- Advanced neural networks.

RSGISLib and associated tools can provide options for applying most of these classifiers through the range of Python libraries which implement these algorithm using numpy (<http://www.numpy.org>) as the array data representation.

Another common difference in approaches is what you classify. Most commonly, classification is undertaken on individual pixels however this is also a wide body of literature which proposes that if a good image segmentation can be achieved, providing a good representation of the environment and features which are being classified then a better overall classification can be achieved. RSGISLib therefore provides
tools for performing per-pixel and segment (object) based classification. In this course both pixel and object methods will be demonstrated using machine learning classifiers from the scikit-learn library.

The scene you are classifying is the Sundarbans on the Indian and Bangladesh border. The classes of interest are:

-   Mangroves

-   Water

-   Agriculture

-   Urban

-   Forests/Rural Livings

You have been provided with some training samples as a shapefiles, one for each class. *Please note: The training data provided is not a demonstration of best practice but quickly drawn for illustration purposes.*

To edit the training samples use QGIS, a video of how to do this is available here:

- https://www.youtube.com/watch?v=EGiR2cM9mIQ

Pixel Based
-----------

Create a new directory `Exercise4` and copy the previous files into that directory including the shapefiles provided to train the classifier. 

The steps for the script to perform the classification are:

-   Rasterise the vector layers providing the training regions for the classification.
    
-   Sample the pixels within the raster regions for each class (i.e., normalise the number of training samples for each class).
    
-   Define the layers to be used for the classification and for which the training data needs to be extracted.
    
-   Extract the training data from the input images for the sampled training pixels and save as a HDF5 file.
    
-   Define the classification classes with the HDF5 file holding the training data and the colour to be used to visualise the classification.
    
-   Create the scikit-learn classifier -- any classifier in the library can be defined with the required parameters.
    
-   Note. the function `classimgutils.findClassifierParametersAndTrain` can be used to find the optimal classifier parameters and train the classifier.
    
-   Train the classifier

-   Apply the classifier

This has been implemented in the following script (`14_PixelBasedClass.py`), copy it into your working directory and run it to produce your classification.

``` python
#/usr/bin/env python

import rsgislib
from rsgislib import imageutils
from rsgislib import vectorutils
import rsgislib.classification
from rsgislib.classification import ClassVecSamplesInfoObj
from rsgislib.classification import classsklearn
from sklearn.ensemble import ExtraTreesClassifier

landsatImg = 'Sundarbans_201511_Landsat_sub_msk.kea'
palsar2Img = 'Sundarbans_15_MOS_F02DAR_utm45n_sub_msk_HHHV_lee_dB.kea'
landsatNDVI = 'Sundarbans_201511_Landsat_sub_msk_ndvi.kea'
landsatWBI = 'Sundarbans_201511_Landsat_sub_msk_wbi.kea'
validImgMsk = 'Sundarbans_validmsk.kea'

mangroveVecFile = 'Mangroves.shp'
mangroveVecLyr = 'Mangroves'
urbanVecFile = 'Urban.shp'
urbanVecLyr = 'Urban'
agricultureVecFile = 'Agriculture.shp'
agricultureVecLyr = 'Agriculture'
forestRuralVecFile = 'ForestRuralUrban.shp'
forestRuralVecLyr = 'ForestRuralUrban'
waterVecFile = 'Water.shp'
waterVecLyr = 'Water'

mangrovePxlTrainVals = 'MangrovesPxlTrainVals.h5'
urbanPxlTrainVals = 'UrbanPxlTrainVals.h5'
agriculturePxlTrainVals = 'AgriculturePxlTrainVals.h5'
forestRuralPxlTrainVals = 'ForestRuralUrbanPxlTrainVals.h5'
waterPxlTrainVals = 'WaterPxlTrainVals.h5'

tmp_path = './tmp'

# The number of pixels to be sampled from each training class
nSamples = 40000

# Define the layers to be used for classification and the bands in those layers.
imgFileInfo = [imageutils.ImageBandInfo(landsatImg, 'landsat', [2,3,4,5,6,7]), 
               imageutils.ImageBandInfo(palsar2Img, 'palsar2', [1,2]), 
               imageutils.ImageBandInfo(landsatNDVI, 'ndvi', [1]), 
               imageutils.ImageBandInfo(landsatWBI, 'wbi', [1])]

# Define the classes and samples for the classification
class_samp_info = list()
class_samp_info.append(ClassVecSamplesInfoObj(id=1, classname='Mangroves', 
                                              vecfile=mangroveVecFile, 
                                              veclyr=mangroveVecLyr, 
                                              fileH5=mangrovePxlTrainVals))
class_samp_info.append(ClassVecSamplesInfoObj(id=2, classname='Urban',  
                                              vecfile=urbanVecFile, 
                                              veclyr=urbanVecLyr, 
                                              fileH5=urbanPxlTrainVals))
class_samp_info.append(ClassVecSamplesInfoObj(id=3, classname='Agriculture', 
                                              vecfile=agricultureVecFile, 
                                              veclyr=agricultureVecLyr, 
                                              fileH5=agriculturePxlTrainVals))
class_samp_info.append(ClassVecSamplesInfoObj(id=4, classname='ForestRural',  
                                              vecfile=forestRuralVecFile, 
                                              veclyr=forestRuralVecLyr, 
                                              fileH5=forestRuralPxlTrainVals))
class_samp_info.append(ClassVecSamplesInfoObj(id=5, classname='Water', 
                                              vecfile=waterVecFile, 
                                              veclyr=waterVecLyr, 
                                              fileH5=waterPxlTrainVals))

# Function which extracts the training data using the defined samples.
classTrainInfo = rsgislib.classification.get_class_training_data(imgFileInfo, 
                                                                 class_samp_info, 
                                                                 tmp_path, nSamples, 
                                                                 validImgMsk)

# Define final colours rather than the randomly allocated from get_class_training_data.
classTrainInfo['Mangroves'].red = 0
classTrainInfo['Mangroves'].green = 153
classTrainInfo['Mangroves'].red = 0

classTrainInfo['Urban'].red = 192
classTrainInfo['Urban'].green = 192
classTrainInfo['Urban'].red = 192

classTrainInfo['Agriculture'].red = 204
classTrainInfo['Agriculture'].green = 204
classTrainInfo['Agriculture'].red = 0

classTrainInfo['ForestRural'].red = 0
classTrainInfo['ForestRural'].green = 204
classTrainInfo['ForestRural'].red = 204

classTrainInfo['Water'].red = 0
classTrainInfo['Water'].green = 0
classTrainInfo['Water'].red = 204


# Create scikit-learn classifier (can be any in the library)
skClassifier = ExtraTreesClassifier(n_estimators=100)
# Train that classifer
classsklearn.train_sklearn_classifier(classTrainInfo, skClassifier)

# Apply the classification.
outImgClass = 'Sundarbans_PxlBaseClass.kea'
classsklearn.apply_sklearn_classifer(classTrainInfo, skClassifier, validImgMsk, 1, 
                                     imgFileInfo, outImgClass, 'KEA')

```

You can see this script being executed in the following video:

- https://www.youtube.com/watch?v=xPAzNCxr8wA

If you need to make some disk space available before going to the next classification stage, remove all the files other than the outputted classification (`Sundarbans_PxlBaseClass.kea`) as we will compare this classification to the object based result produced in the next step.

Object Based
------------

Object based classification first required that a segmentation is undertaken. The algorithm provided in RSGISLib is executed using the function `segutils.runShepherdSegmentation`. The algorithm is initialised using a K-Means clustering and then small objects are eliminated, merging into their 'nearest' neighbour (defined using the euclidean distance), refer to Shepherd et al., 2019 and Clewley et al., 2014 for more details.

The segmentation image (`Sundarbans_Clumps.kea`) has been provided to save analysis time but is generated using the following script (`15_SegData.py`):

``` python
#/usr/bin/env python

import rsgislib 
from rsgislib.segmentation import segutils

landsatImg = 'Sundarbans_201511_Landsat_sub_msk.kea'
palsar2Img = 'Sundarbans_15_MOS_F02DAR_utm45n_sub_msk_HHHV_lee.kea'

# Select the image bands to be used for segmentation from the landsat image.
landsatImg564 = 'Sundarbans_201511_Landsat_sub_msk_b564.kea'
rsgislib.imageutils.selectImageBands(landsatImg, landsatImg564, 
                                     'KEA', rsgislib.TYPE_16UINT, [5,6,4])

# Stack the selected landsat bands with the PALSAR-2 imagery.
segTmpImg = 'Sundarbans_SegTempImg.kea'
rsgislib.imageutils.stackImageBands([landsatImg564,palsar2Img], None, 
                                    segTmpImg, None, 0, 'KEA', 
                                    rsgislib.TYPE_32FLOAT)

# Perform the segmentation
clumpsImg = 'Sundarbans_Clumps.kea'
segutils.runShepherdSegmentation(segTmpImg, clumpsImg, tmpath='./segtmp', 
                                 numClusters=120, minPxls=50, distThres=100, 
                                 sampling=100, kmMaxIter=200)

```

If helpful, you can watch a video of this code being executed here:

- https://www.youtube.com/watch?v=q79ji9S4jNo

For very large image datasets there is a tiled version of the segmentation algorithm. At the moment, this is still single threaded but in the future a version of the algorithm will be able to use multiple processing cores on a single machine. The function is  `rsgislib.segmentation.tiledsegsingle.performTiledSegmentation`.

For the next stage of processing you will need to create a new working directory (`Exercise5`), copy the files produced during the pre-processing other than the dB images (we calculate the mean of the power image and then convert to dBs) and the clumps image (`Sundarbans_Clumps.kea`) into that working directory.

The stages of processing which will be undertaken to perform the classification are as follows (within `16_PerformObjClass.py`)

-   Populate the segmentation (clumps file) with the variables you wish to use for the classification. Building the raster attribute table. For this study we will populate with:
    
    -   Mean HH and HV backscatter (power)

    -   Mean Landsat reflectance

    -   Mean NDVI

    -   Mean WBI

-   Convert mean HH and HV power values to dBs.

-   Define the training data and add it to the RAT.

-   Balance the training data so the number of samples are similar/same for each class.
    
-   Define variables to be used for the classification

-   Search for the optimal classifier parameters (Grid Search) and train
    the classifier.

-   Define the class output colours

-   Applying the classifier

-   Collapse the clumps file to just the classification

This is implemented with the following script, copy it into your working directory and run it.

``` python
#/usr/bin/env python

from rsgislib import rastergis
from rsgislib.rastergis import ratutils
from rsgislib.classification import classratutils
from rsgislib import classification

import osgeo.gdal as gdal
from rios import rat
import numpy

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV

landsatImg = 'Sundarbans_201511_Landsat_sub_msk.kea'
palsar2Img = 'Sundarbans_15_MOS_F02DAR_utm45n_sub_msk_HHHV.kea'
landsatNDVI = 'Sundarbans_201511_Landsat_sub_msk_ndvi.kea'
landsatWBI = 'Sundarbans_201511_Landsat_sub_msk_wbi.kea'
validImgMsk = 'Sundarbans_validmsk.kea'

clumpsImg ='Sundarbans_Clumps.kea'

# Add PALSAR-2 HH and HV Mean power to each of the segments
bs = [rastergis.BandAttStats(band=1, meanField='HHPow'),
      rastergis.BandAttStats(band=2, meanField='HVPow')]
rastergis.populateRATWithStats(palsar2Img, clumpsImg, bs)

# Add mean landsat reflectance to each of the segments
bs = [rastergis.BandAttStats(band=1, meanField='CoastMean'),
      rastergis.BandAttStats(band=2, meanField='BlueMean'),
      rastergis.BandAttStats(band=3, meanField='GreenMean'),
      rastergis.BandAttStats(band=4, meanField='RedMean'),
      rastergis.BandAttStats(band=5, meanField='NIRMean'),
      rastergis.BandAttStats(band=6, meanField='SWIR1Mean'),
      rastergis.BandAttStats(band=7, meanField='SWIR2Mean')]
rastergis.populateRATWithStats(landsatImg, clumpsImg, bs)

# Add mean NDVI reflectance to each of the segments
bs = [rastergis.BandAttStats(band=1, meanField='NDVI')]
rastergis.populateRATWithStats(landsatNDVI, clumpsImg, bs)

# Add mean NDVI reflectance to each of the segments
bs = [rastergis.BandAttStats(band=1, meanField='WBI')]
rastergis.populateRATWithStats(landsatWBI, clumpsImg, bs)

# Create and Calculate dB columns
ratDataset = gdal.Open(clumpsImg, gdal.GA_Update)
HHPow = rat.readColumn(ratDataset, 'HHPow')
HVPow = rat.readColumn(ratDataset, 'HVPow')
 
HHdB = numpy.where(HHPow>0, 10*numpy.log10(HHPow), 0)
HVdB = numpy.where(HVPow>0, 10*numpy.log10(HVPow), 0)
 
rat.writeColumn(ratDataset, "HHdB", HHdB)
rat.writeColumn(ratDataset, "HVdB", HVdB)
ratDataset = None

# Create list of training data and populate the RAT.
classesDict = dict()
classesDict['Mangroves'] = [1, 'Mangroves.shp']
classesDict['Urban'] = [2, 'Urban.shp']
classesDict['Agriculture'] = [3, 'Agriculture.shp']
classesDict['ForestRural'] = [4, 'ForestRuralUrban.shp']
classesDict['Water'] = [5, 'Water.shp']
tmpPath = './tmp'
classesIntColIn = 'ClassInt'
classesNameCol = 'ClassStr'
ratutils.populateClumpsWithClassTraining(clumpsImg, classesDict, tmpPath, 
                                          classesIntColIn, classesNameCol)
 
# Ensure there are a minimum number of training samples and 
# balance so there are the same number for each class
classesIntCol = 'ClassIntSamp'
classratutils.balanceSampleTrainingRandom(clumpsImg, classesIntColIn, 
                                           classesIntCol, 100, 200)

# RAT variables used for classification
variables = ['HHdB', 'HVdB', 'NIRMean', 'SWIR1Mean', 'RedMean', 'NDVI', 'WBI']


classParameters = {'n_estimators':[10,100,500], 'max_features':[2,3,4]}
gSearch = GridSearchCV(ExtraTreesClassifier(), classParameters)
preProcessing = None
classifier = classratutils.findClassifierParameters(clumpsImg, classesIntCol, variables, 
                                                    preProcessor=None, gridSearch=gSearch)

classColours = dict()
classColours['Mangroves'] = [0,153,0]
classColours['Urban'] = [192,192,192]
classColours['Agriculture'] = [204,204,0]
classColours['ForestRural'] = [0,204,204]
classColours['Water'] = [0,0,204]
 
# Perform classification...
outClassIntCol = 'OutClass'
outClassStrCol = 'OutClassName'
classratutils.classifyWithinRAT(clumpsImg, classesIntCol, classesNameCol, variables,
                                classifier=classifier, outColInt=outClassIntCol,
                                outColStr=outClassStrCol, classColours=classColours,
                                preProcessor=preProcessing)
  
# Export to a 'classification image' rather than a RAT...
outClassImg = 'Sundarbans_ObjBaseClass.kea'
classification.collapseClasses(clumpsImg, outClassImg, 'KEA', 'OutClassName', 'OutClass')

```

If helpful, you can watch this script being executed in the following video:

- https://www.youtube.com/watch?v=fCu5c1__jQE

Compare the Classifications
---------------------------

You have now produced two classification through different methods but using the same data, classifier and training data. How do they compare?

See a video of using Tuiview to compare the classifications here:

- https://www.youtube.com/watch?v=IuS0aNm2Two

Conclusions
===========

I hope following the completion of this course you feel you have a better idea of how you can use RSGISLib for your spatial data processing. This course has covered some of the most commonly used functions within RSGISLib however there are 100s more functions you can explore and try out -- see the website. Particular areas we have not covered include:

-   `Image Calibration` -- tools for calibrating imagery, primarily optical data and used within the ARCSI
    ([www.rsgislib.org/arcsi](www.rsgislib.org/arcsi)) software for performing atmospheric correction of optical satellite imagery.
    
-   `Elevation` -- tools for doing things with DEMs.

-   `Image Morphology` -- functions for applying image morphology operations
    
-   `Image Registration` -- function for applying an automated image-to-image registration
    
-   `Zonal Statistics` -- functions for retrieving information into a vector attribute table from a raster image.

We've only scratched the surface of a number of libraries (i.e., `rastergis`) and I would recommend you have a look though these on the website <http://www.rsgislib.org>.

Don't forget, that any or all of these scripts can be wrapped up as functions and command line tools if required or even attached to a GUI using pyqt or similar library for creating your GUI. If wrapped up as
functions then don't forget that you can use the multiprocessor module to apply batch processing using multiple processing cores.



Advanced Options: Multiple Cores and Command line tools
=======================================================

Using multiple cores
--------------------

An extension for more advanced users is to use the Python multiprocessing module
(<https://docs.python.org/3.7/library/multiprocessing.html>) to enable multi-core processing. We'll use the `Pool` object which requires a function to do 'the work', where the function takes just a single parameter (which can be an array).

Therefore, we will edit `02B_PALSARStack.py`, creating`02C_PALSARStack.py` so the three parameters are passed as an array:

``` python
# Old functions
def createPALSARStack(inputTAR, outputStackImg, tmpDir):

# New function interface
def createPALSARStack(params):
    inputTAR = params[0]
    outputStackImg = params[1]
    tmpDir = params[2]
```

We then define the input parameters as an array of arrays, find the number of processing cores available (set to an integer > 0 and < the total number of cores if you don't want all to be used). A pool is then
created and the list of parameters queued and run on the available cores.

``` python
inputParams = []
inputParams.append(['N22E088_15_MOS_F02DAR.tar.gz', 'N22E088_15_MOS_F02DAR_Stack.kea', './tmp'])
inputParams.append(['N22E089_15_MOS_F02DAR.tar.gz', 'N22E089_15_MOS_F02DAR_Stack.kea', './tmp'])
inputParams.append(['N23E088_15_MOS_F02DAR.tar.gz', 'N23E088_15_MOS_F02DAR_Stack.kea', './tmp'])
inputParams.append(['N23E089_15_MOS_F02DAR.tar.gz', 'N23E089_15_MOS_F02DAR_Stack.kea', './tmp'])

numCores = multiprocessing.cpu_count() # find the number of cores available on the system.
with multiprocessing.Pool(numCores) as p:
    p.map(createPALSARStack, inputParams)
```

Look at the code in `02C_PALSARStack.py`, try to understand what is going, and then copy the file into your `Exercise2` removing the previously created stacked files and execute it as you did before the result will be the same as `02B_PALSARStack.py` but will execute faster as multiple images will be processing at the same time on different cores.

``` bash
python 02C_PALSARStack.py
```

Make a command line tool
------------------------

Building on script `02B_PALSARStack.py` the next step will be to make this tool parametrisable by the user from the terminal. Python has an easy to use module, `argparse` (<https://docs.python.org/3/library/argparse.html>), for this purpose.

We will just change the bottom of the script to read the three parameters from the terminal:

``` python
# Only run this code if it is called from the terminal (i.e., not from another python script)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, type=str, 
                        help="""Input tar.gz file""")
    parser.add_argument("-o", "--output", required=True, type=str, 
                        help="""Output stacked KEA file""")
    parser.add_argument("-t", "--tmpath", required=True, type=str, 
                        help="""Temporary path which will be generated and 
                                removed during processing.""")
                        
    args = parser.parse_args()
    
    createPALSARStack(args.input, args.output, args.tmpath)
```

You can now run this script from the terminal (from your `Exercise2` directory) using the following command:

``` bash
python 02D_PALSARStack.py -i N22E088_15_MOS_F02DAR.tar.gz -o N22E088_Stack.kea -t tmp
```

If you wanted your script to process a whole directory of tar.gz files then you can make the command line parser code as follows (`02E_PALSARStack.py`):

``` python
# Only run this code if it is called from the terminal (i.e., not from another python script)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, type=str, 
                        help="""Input directory containing the tar.gz files""")
    parser.add_argument("-o", "--output", required=True, type=str, 
                        help="""Output directory for the KEA files""")
    parser.add_argument("-t", "--tmpath", required=True, type=str, 
                        help="""Temporary path which will be generated and 
                                removed during processing.""")
                        
    args = parser.parse_args()
    
    # Get all the tar.gz files.
    inputFiles = glob.glob(os.path.join(args.input, '*.tar.gz'))
    
    # Loop through files.
    for inFile in inputFiles:
        # Get basename
        basename = os.path.basename(inFile).split('.')[0]
        # Create name of output file.
        outputFile = os.path.join(args.output, basename+'_stack.kea')
        # Call function.
        createPALSARStack(inFile, outputFile, args.tmpath)
```

You can now run this script from the terminal (from your `Exercise2` directory) using the following command:

``` bash
python 02E_PALSARStack.py -i . -o . -t tmp
```

NOTE. '.' signifies the current directory in your terminal. Try running`ls .`

Finally, we can merge `02E_PALSARStack.py` with `02C_PALSARStack.py` to allow multiple cores to be used.

``` python
# Only run this code if it is called from the terminal (i.e., not from another python script)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, type=str, 
                        help="""Input directory containing the tar.gz files""")
    parser.add_argument("-o", "--output", required=True, type=str, 
                        help="""Output directory for the KEA files""")
    parser.add_argument("-t", "--tmpath", required=True, type=str, 
                        help="""Temporary path which will be generated and 
                                removed during processing.""")
    args = parser.parse_args()
    
    # Get all the tar.gz files.
    inputFiles = glob.glob(os.path.join(args.input, '*.tar.gz'))
    # Array for the function parameters.
    inputParams = []
    # Loop through files.
    for inFile in inputFiles:
        # Get basename
        basename = os.path.basename(inFile).split('.')[0]
        # Create name of output file.
        outputFile = os.path.join(args.output, basename+'_stack.kea')
        # Add parameters to array.
        inputParams.append([inFile, outputFile, args.tmpath])
        
    # find the number of cores available on the system.
    numCores = multiprocessing.cpu_count()
    # Create pool and run.
    with multiprocessing.Pool(numCores) as p:
        p.map(createPALSARStack, inputParams)
```

Try executing `02F_PALSARStack.py`, from your `Exercise2` directory, and you find all the files processing. This script could be used to processing thousands of files if required using all the cores on your system.

``` bash
python 02F_PALSARStack.py -i . -o . -t tmp
```
