title: RSGISLib Introduction to Classification Tutorial
category: Earth Observation
tags: Python, RSGISLib, Image Processing.
date: 2020-02-04
modified: 2020-02-04


------------------------------------------------------------------------

*This work (including scripts) is licensed under a Creative Commons Attribution 4.0 International License. To view a copy of this license, visit <http://creativecommons.org/licenses/by/4.0/>. With the exception of Sentinel-2 imagery which is copyright © ESA.*

------------------------------------------------------------------------


## Overview

This tutorial will take you through many of the classification functions within the RSGISLib software. This tutorial will also make use of the following machine learning libraries:

- Scikit-Learn
- LightGBM
- Keras / Tensorflow (deep learning)

Following the completion of this tutorial it is expected that you will be able:

1. Undertake a classification within on EO data using the RSGISLib library.
2. Optimise the parameters used for the classification.
3. Compare the classification results from different classifiers and select the most appropriate.
4. Undertake an appropriate accuracy assessment following standard methods

## Download Datasets

The datasets and scripts needed for this worksheet can be downloaded from the link below:

- [https://www.remotesensing.info/downloads/rsgislib_adv_class_tutorial.tar.gz](https://www.remotesensing.info/downloads/rsgislib_adv_class_tutorial.tar.gz)


## Machine Learning Libraries

### Scikit-Learn
The scikit-learn library is the most commonly used machine learning library in Python with access to many classification, clustering and regression algorithms. This includes commonly used algorithms such as:

 - Random Forests
 - Support Vector Machines
 - Neural Networks

### LightGBM
The LightGBM library is primarily focused on the application of Gradient Boosted Decision Trees. Boosting is a process by which classifiers are layered on each other with the second trained to classify those samples which the first miss classifies. 

### Keras / Tensorflow
Keras and Tensorflow are focused on the creation and application of Neural Networks. Specifically, the creation of large complex networks which are commonly referred to as deep learning classifiers.

## Sentinel-2 Imagery

For this exercise you will use a Sentinel-2 scene from the Aberystwyth area from 29th June 2018 (Figure 1). This image has been processed using the ARCSI software ([https://arcsi.remotesensing.info](https://arcsi.remotesensing.info)) to derive a surface reflectance product that has also been topographically corrected. It should be noted that the summer of 2018 was particularly dry and therefore many of the grasslands have a low NIR reflectance with a large amount of non-photosynthetic vegetation.

<img src="figures/sen2_scene.png" alt="sen2_scene" style="zoom:67%;" />

Figure 1. Sentinel-2 Scene from 29/06/2018.

## Define Training data

You have been provided with training for the following classes:

- Conifer Forest
- Deciduous Forest
- Water
- Artificial Surfaces (i.e., Urban)
- Non-Photosynthetic Vegetation
- Bare Rock and Sand
- Scrub
- Bracken
- Grass Short
- Grass Long

These are within the `aber_sen2_cls_training.gpkg` file as separate layers.

## Undertaking the tutorial

The tutorial has provided a download with the datasets and scripts (`rsgislib_adv_class_tutorial.tar.gz`). To start the tutorial you should extract that archive:

```bash
tar -zxf rsgislib_adv_class_tutorial.tar.gz
```

You will then have `dataset` and `scripts` directories. These contain all you need to run this tutorial and it is expected that all commands and scripts are executed from within the `scripts` directory. To change to the scripts directory run the command:

```bash
cd rsgislib_adv_class_tutorial/scripts
```

If you would like to 'cheat' and get the outputs for some of the scripts which take a longer time to execute (e.g., the KNN classification) then you can download the `rsgislib_adv_class_tutorial_prods.tar.gz`file, within which all the output files are provided.

## Exercise 1. Compare the classifiers from Scikit-Learn
Within this exercise you will need to compare the following classifiers within scikit-learn:

1. Maximum Likelihood
2. Support Vector Machines (SVM)
3. K- Nearest Neighbour (KNN)
4. Random Forests
5. Extremely Randomized Trees
6. Gradient Boosted Decision Trees
7. Neural Network

### Prepare the data
Some classifiers (e.g., Support Vector Machines) require the image pixel values to be normalised so they are within the same range. If you were merging data from different modalities, this would be even more important (e.g., dB values from SAR are negative). Within the Sentinel-2 data the range of values for each band can be quite different, for example in the visible bands range is commonly quite low, while in the near infrared (NIR) the range is high. 

There are different approaches to normalising the data but for this tutorial we will try two:

- minimum -- maximum normalisation
- standard deviation normalisation

Following the application of the normalisation to the input imagery, the image pixel values will be extracted from the images. This will result in three training sets, the two normalised and original datasets.

#### Min-Max Normalisation
Applied on a per-band basis this normalisation calculates the minimum and maximum pixel values and then uses those to scale the rest of the data to the same range:

$$
out_X = \frac{X - min}{max-min} \times out_{range}
$$
Where $X$ is the current pixel, $min$ is the minimum for the whole image band, $max$ is the maximum for the whole image band, $out_{range}$ is the maximum image pixel value within the output image and $out_X$ is the output image pixel value written to the output image.

#### Standard Deviation Normalisation
Also applied on a per-band basis, this normalisation calculates the $mean$ and standard deviation ($stdev$) for each image band. The user provides the number of standard deviations ($n_{userstdevs}$) the data should be normalised over (e.g., 2 standard deviations). This provides the upper ($up_{stdev}$) and lower ($low_{stdev}$) bounds for the normalisations. 

$$
low_{stdev} = mean - (std \times n_{userstdevs}) \\
\text{if } min > low_{stdev} \text{ then } low_{stdev} = min \\
up_{stdev} = mean + (std \times n_{userstdevs}) \\
\text{if } max < up_{stdev} \text{ then } up_{stdev} = max \\
out_X = \frac{X - low_{stdev}}{up_{std}-low_{stdev}} \times out_{range}
$$
Where $X$ is the current pixel, $min$ is the minimum for the whole image band, $max$ is the maximum for the whole image band, $out_{range}$ is the maximum image pixel value within the output image and $out_X$ is the output image pixel value written to the output image.

### Normalising the Image

To apply the normalisation run the script `01_image_normalisation.py` from within the scripts directory. 

```bash
python 01_image_normalisation.py
```

The script will save the outputs as GeoTIFF files with compression, using the LZW algorithm. This requires the `RSGISLIB_IMG_CRT_OPTS_GTIFF` environmental variable to be set. 

```python
import rsgislib
import rsgislib.imageutils
from rsgislib.imageutils import STRETCH_LINEARMINMAX
from rsgislib.imageutils import STRETCH_LINEARSTDDEV
import os

# Define environmental variable so outputted GeoTIFFs are tiled and compressed.
os.environ["RSGISLIB_IMG_CRT_OPTS_GTIFF"] = "TILED=YES:COMPRESS=LZW:BIGTIFF=YES"

# The input image
input_img = '../dataset/SEN2_20180629_T30UVD_ORB037_osgb_stdsref.tif'

# The output image file for the linear normalisation
output_lin_img = '../dataset/SEN2_20180629_T30UVD_ORB037_osgb_stdsref_norm_linear.tif'
# Run the linear normalisation
rsgislib.imageutils.normaliseImagePxlVals(input_img, output_lin_img, 'GTIFF', 
                                          rsgislib.TYPE_16UINT, innodataval=0, 
                                          outnodataval=0, outmin=0, outmax=1000, 
                                          stretchtype=STRETCH_LINEARMINMAX)
# Calculate image statistics and pyramids for the output image
rsgislib.imageutils.popImageStats(output_lin_img, usenodataval=True, 
                                  nodataval=0, calcpyramids=True)

# The output image file for the standard deviation normalisation
output_sd_img = '../dataset/SEN2_20180629_T30UVD_ORB037_osgb_stdsref_norm_stddev.tif'
# Run the standard deviation normalisation
rsgislib.imageutils.normaliseImagePxlVals(input_img, output_sd_img, 'GTIFF', 
                                          rsgislib.TYPE_16UINT, innodataval=0, 
                                          outnodataval=0, outmin=0, outmax=1000, 
                                          stretchtype=STRETCH_LINEARSTDDEV, 
                                          stretchparam=2)
# Calculate image statistics and pyramids for the output image
rsgislib.imageutils.popImageStats(output_sd_img, usenodataval=True, 
                                  nodataval=0, calcpyramids=True)

```

Now, have a look at the resulting images within tuiview with the stretch set as linear to see the difference between the images. Also, look at the pixel values using the query tool.

```bash
tuiview ../dataset/*.tif
```

### Extract Training Data

To extract the training data you need to run the script `02_extract_training_data.py`, which is shown below. For the three input images (reflectance, linear normalisation and standard deviation normalisation) the script

- Rasterise the training samples
- Extracts the image pixel values to a HDF5 file.
- Samples the HDF5 file to get the training dataset (this balances the samples so you have the same number of samples per class).
- Sampling the remaining samples to get the testing dataset. 

The classification samples are provides as a vector file, in GeoPackage format, this allows each of the classes to be represented by a different vector layer while keeping things effecient with just a single file.

```python
import rsgislib
import rsgislib.vectorutils
import rsgislib.imageutils

import os.path

# The input image files
refl_img_file = '../dataset/SEN2_20180629_T30UVD_ORB037_osgb_stdsref.kea'
norm_lin_img = '../dataset/SEN2_20180629_T30UVD_ORB037_osgb_stdsref_norm_linear.tif'
norm_sd_img = '../dataset/SEN2_20180629_T30UVD_ORB037_osgb_stdsref_norm_stddev.tif'

# The input training data vector file (each class if different layer)
train_vec_file = '../dataset/aber_sen2_cls_training.gpkg'

# The output directory where the training will be outputted
out_dir = '../dataset/training'

# A temp directory where intermeditate files will be outputted
# can be deleted after processing has finished.
tmpdir = './tmp'

# Create a unique string to be appended to run folder name
# so each time the script runs it does not over write the 
# intermeditate from a previous run.
rsgis_utils = rsgislib.RSGISPyUtils()
uid_str = rsgis_utils.uidGenerator()

# Check the temp directory is present.
if not os.path.exists(tmpdir):
    raise Exception("Temporary Directory does not exist, please create.")

# Check the output directory is present
if not os.path.exists(out_dir):
    raise Exception("Output Training Directory does not exist, please create.")

# Create the temp directory for this run.
c_tmp_dir = os.path.join(tmpdir, 'extract_training_{}'.format(uid_str))
os.mkdir(c_tmp_dir)

# Get a list of all the layers within the vector file
# i.e., a list of all the classes.
lyr_lst = rsgislib.vectorutils.getVecLyrsLst(train_vec_file)

# The number of training samples for each class
n_train_samples = 1000
# The number of testing samples for each class
n_test_samples = 400
# The number of validate samples for each class
n_valid_samples = 400

# Iterate through the layers
for lyr_name in lyr_lst:
    print(lyr_name)
    
    # Rasterise the training samples.
    samples_img = os.path.join(c_tmp_dir, "samples_{}.kea".format(lyr_name))
    rsgislib.vectorutils.rasteriseVecLyr(train_vec_file, lyr_name, refl_img_file, 
                                         samples_img, gdalformat='KEA', burnVal=1)
    
    #############################
    # Get the refl image samples, save as HDF5 file.
    refl_samples_h5 = os.path.join(c_tmp_dir, "refl_samples_{}.h5".format(lyr_name))
    rsgislib.imageutils.extractZoneImageValues2HDF(refl_img_file, samples_img, 
                                                   refl_samples_h5, 1)
    
    refl_train_samples_h5 = os.path.join(out_dir,
                                         "refl_train_samples_{}.h5".format(lyr_name))
    refl_test_samples_h5 = os.path.join(out_dir, 
                                        "refl_test_samples_{}.h5".format(lyr_name))
    refl_valid_samples_h5 = os.path.join(out_dir, 
                                        "refl_valid_samples_{}.h5".format(lyr_name))
    
    rsgislib.classification.split_sample_train_valid_test(refl_samples_h5, 
                                                          refl_train_samples_h5, 
                                                          refl_valid_samples_h5, 
                                                          refl_test_samples_h5, 
                                                          n_test_samples, 
                                                          n_valid_samples, 
                                                          train_sample=n_train_samples, 
                                                          rand_seed=42)
    #############################
    
    #############################
    # Get the Norm Linear samples, save as HDF5 file.
    normlin_samples_h5 = os.path.join(c_tmp_dir, "normlin_samples_{}.h5".format(lyr_name))
    rsgislib.imageutils.extractZoneImageValues2HDF(norm_lin_img, samples_img, 
                                                   normlin_samples_h5, 1)
    
    normlin_train_samples_h5 = os.path.join(out_dir,
                                            "normlin_train_samples_{}.h5".format(lyr_name))
    normlin_test_samples_h5 = os.path.join(out_dir, 
                                           "normlin_test_samples_{}.h5".format(lyr_name))
    normlin_valid_samples_h5 = os.path.join(out_dir, 
                                            "normlin_valid_samples_{}.h5".format(lyr_name))
    
    rsgislib.classification.split_sample_train_valid_test(normlin_samples_h5, 
                                                          normlin_train_samples_h5, 
                                                          normlin_valid_samples_h5, 
                                                          normlin_test_samples_h5, 
                                                          n_test_samples, 
                                                          n_valid_samples, 
                                                          train_sample=n_train_samples, 
                                                          rand_seed=42)
    #############################
    
    #############################
    # Get the Norm StdDev samples, save as HDF5 file.
    normsd_samples_h5 = os.path.join(c_tmp_dir, "normsd_samples_{}.h5".format(lyr_name))
    rsgislib.imageutils.extractZoneImageValues2HDF(norm_sd_img, samples_img, 
                                                   normsd_samples_h5, 1)
    
    normsd_train_samples_h5 = os.path.join(out_dir,
                                            "normsd_train_samples_{}.h5".format(lyr_name))
    normsd_test_samples_h5 = os.path.join(out_dir, 
                                           "normsd_test_samples_{}.h5".format(lyr_name))
    normsd_valid_samples_h5 = os.path.join(out_dir, 
                                            "normsd_valid_samples_{}.h5".format(lyr_name))
    
    rsgislib.classification.split_sample_train_valid_test(normsd_samples_h5, 
                                                          normsd_train_samples_h5, 
                                                          normsd_valid_samples_h5, 
                                                          normsd_test_samples_h5, 
                                                          n_test_samples, 
                                                          n_valid_samples, 
                                                          train_sample=n_train_samples, 
                                                          rand_seed=42)
    #############################

```



### Applying the Classifiers

The scripts for applying the classifiers are all very similar with just the names of the output files and the classifier being changed. The classification is performed on all three datasets (original reflectance, linear normalisation, standard deviation normalisation), for each classification the steps within the script are:

- Define the image info list, using the `rsgislib.imageutils.ImageBandInfo` object. In this case, we just have one image and we are using all the image bands.
- Create an instance of the scikit-learn classifier. You would specify any parameters you wished to use at this point.
- Define the class training info, using the `rsgislib.classification.ClassSimpleInfoObj` object. This is provided an a dictionary with the key being the name of the class and then the `rsgislib.classification.ClassSimpleInfoObj` instance specifies the output pixel value (`id`), the training data as a HDF5 and the RGB colour of the class in the output image.
- Train the classifier.
- Apply the classifier.

Before running the classification you will need to create an output directory for the classifications, within the script this is specified as `../dataset/cls_results` you can create this directory with the following command, run from within the scripts directory:

```bash
mkdir ../dataset/cls_results
```

The scripts have been provided for all the classifiers try running and following and reading the documentation (share the facts you learn about these classifiers with others):

- `python 03a_apply_max_likelihood.py`
  - Documentation: [https://scikit-learn.org/stable/modules/naive_bayes.html#gaussian-naive-bayes](https://scikit-learn.org/stable/modules/naive_bayes.html#gaussian-naive-bayes)
- `python 03b_apply_svm.py`
  - Documentation: [https://scikit-learn.org/stable/modules/svm.html#svm-classification](https://scikit-learn.org/stable/modules/svm.html#svm-classification)
- `python 03c_apply_knn.py` (note might take some time to run)
  - Documentation: [https://scikit-learn.org/stable/modules/neighbors.html#classification](https://scikit-learn.org/stable/modules/neighbors.html#classification)
- `python 03d_apply_random_forests.py`
  - Documentation: [https://scikit-learn.org/stable/modules/ensemble.html#forest](https://scikit-learn.org/stable/modules/ensemble.html#forest)
- `python 03e_apply_extra_trees.py`
  - Documentation: [https://scikit-learn.org/stable/modules/ensemble.html#forest](https://scikit-learn.org/stable/modules/ensemble.html#forest)
- `python 03f_apply_gbdt.py`
  - Documentation: [https://scikit-learn.org/stable/modules/ensemble.html#gradient-boosting](https://scikit-learn.org/stable/modules/ensemble.html#gradient-boosting)
- `python 03g_apply_neural_networks.py`
  - Documentation: [https://scikit-learn.org/stable/modules/neural_networks_supervised.html](https://scikit-learn.org/stable/modules/neural_networks_supervised.html)

The script for running the first of the classifications, with the original reflectance data using the random forests classifier is shown below:

```python
import os.path
from rsgislib.imageutils import ImageBandInfo
from rsgislib.classification import ClassSimpleInfoObj
import rsgislib.classification.classsklearn
from sklearn.ensemble import RandomForestClassifier

out_cls_dir = '../dataset/cls_results'
if not os.path.exists(out_cls_dir):
    raise Exception("Output Directory does not exist, please create.")

############################################################################################
# The input image
refl_img_file = '../dataset/SEN2_20180629_T30UVD_ORB037_osgb_stdsref.tif'
imgs_info = []
imgs_info.append(ImageBandInfo(fileName=refl_img_file, name='sen2', 
                               bands=[1,2,3,4,5,6,7,8,9,10]))

valid_img = '../dataset/SEN2_20180629_T30UVD_ORB037_osgb_stdsref_vmsk.kea'

# Apply Random Forests Classifier using the Original Reflectance data.
skclf = RandomForestClassifier(n_estimators=100)

cls_train_info = dict()
cls_train_info['Artificial_Surfaces'] = ClassSimpleInfoObj(id=1, 
            fileH5='../dataset/training/refl_train_samples_Artificial_Surfaces.h5', 
            red=160, green=160, blue=160)
cls_train_info['Bare_Rock_Sand']      = ClassSimpleInfoObj(id=2, 
            fileH5='../dataset/training/refl_train_samples_Bare_Rock_Sand.h5', 
            red=100, green=100, blue=100)
cls_train_info['Bracken']             = ClassSimpleInfoObj(id=3, 
            fileH5='../dataset/training/refl_train_samples_Bracken.h5', 
            red=235, green=146, blue=38)
cls_train_info['Conifer_Forest']      = ClassSimpleInfoObj(id=4, 
            fileH5='../dataset/training/refl_train_samples_Conifer_Forest.h5', 
            red=2, green=97, blue=16)
cls_train_info['Deciduous_Forest']    = ClassSimpleInfoObj(id=5, 
            fileH5='../dataset/training/refl_train_samples_Deciduous_Forest.h5', 
            red=50, green=184, blue=69)
cls_train_info['Grass_Long']          = ClassSimpleInfoObj(id=6, 
            fileH5='../dataset/training/refl_train_samples_Grass_Long.h5', 
            red=150, green=227, blue=18)
cls_train_info['Grass_Short']         = ClassSimpleInfoObj(id=7, 
            fileH5='../dataset/training/refl_train_samples_Grass_Short.h5',
            red=194, green=219, blue=66)
cls_train_info['NonPhoto_Veg']        = ClassSimpleInfoObj(id=8, 
            fileH5='../dataset/training/refl_train_samples_NonPhotosynthetic_Vegetation.h5', 
            red=98, green=225, blue=227)
cls_train_info['Scrub']               = ClassSimpleInfoObj(id=9, 
            fileH5='../dataset/training/refl_train_samples_Scrub.h5', 
            red=0, green=80, blue=0)
cls_train_info['Water']               = ClassSimpleInfoObj(id=10, 
            fileH5='../dataset/training/refl_train_samples_Water_Training.h5', 
            red=0, green=0, blue=255)

# Train the classifier
rsgislib.classification.classsklearn.train_sklearn_classifier(cls_train_info, skclf)

# Apply the classifier to the image
out_cls_img = os.path.join(out_cls_dir, 'cls_rf_refl.kea')
rsgislib.classification.classsklearn.apply_sklearn_classifer(cls_train_info, skclf, 
                                                             valid_img, 1, imgs_info, 
                                                             out_cls_img, 'KEA', 
                                                             classClrNames=True)
############################################################################################

```

You have now produced 21 classifications (19 if the KNN was taking a long time to run), so, which one is best? Which one should you be using for your application?

## Exercise 2. Perform an Accuracy Assessment

### Create Accuracy Assessment Points

To undertake an accuracy assessment you first need to generate a set of points, there are a number of options for doing this:

- Random - n points are randomly placed within the scene.
- Stratified - n points of points per class are placed within the scene.
- Stratified Proportional to Area - the number of points placed within a class are proportional to the area mapped.

In this case we will undertake a stratified assessment, the script below allows this to be calculated. Note. if you had produced a classification using a segmentation (not cover by this tutorial) then you would have to 'collapse' the attribute table to a classification before apply this script.

```python
import rsgislib.classification

in_cls_img = '../dataset/cls_results/cls_rf_refl.kea'
acc_points_shp = '../dataset/acc_points/cls_acc_assessment_pts.shp'
rsgislib.classification.generateStratifiedRandomAccuracyPts(in_cls_img, 
                                            acc_points_shp, 'ClassName', 
                                            'rf_rl_cls', 'ref_pts', 
                                            100, 42, True, True)

```

This script can be executed as:

```bash
python 04_gen_stratified_accpoints.py
```

You should now have a ESRI Shapefile which has an attribute table simlar to the Table below:

| rf_rl_cls           | ref_pts             | Processed |
| ------------------- | ------------------- | --------- |
| Artificial_Surfaces | Artificial_Surfaces | 0         |
| Artificial_Surfaces | Artificial_Surfaces | 0         |
| Deciduous_Forest    | Deciduous_Forest    | 0         |
| ...                 | ...                 | 0         |
| ...                 | ...                 | 0         |
| Water               | Water               | 0         |

The `rf_rl_cls`column is the classification from the image column and the `ref_pts` column will have the reference class for the points. Initially `ref_pts` has been populated with the same value as the `rf_rl_cls` column, this helps to speed up the assignment of `ref_pts` to the correct class used for the accuracy assessment as the majority of the points should already be correct if your map is good. The `Processed` column is to identify whether a point has been processed (i.e., the class checked) or not. A value of `0` is not processed (i.e., the default starting value) and a value of `1` means the point has been checked.

### QGIS Plugin

To facilitate the accuracy assessment, which can be significant amount of work, a QGIS plugin has been written ([https://bitbucket.org/petebunting/classaccuracy](https://bitbucket.org/petebunting/classaccuracy)). The plugin loads the shapefile and you then select the appropriate columns and it then guides you to each point, if the classification is correct you just press the `Enter` key or the 'Next Button', if the classification is incorrect then you can select the correct classification from the drop down box, or use the number keys and then press `Enter` or the 'Next Button' to continue to the next point.

Once you have completed all the points you can then calculate an error matrix from using the plugin. 

#### Installation of the Plugin

To install the plugin download the `ClassAccuracyMain_qgis3.zip` file:

[https://bitbucket.org/petebunting/classaccuracy/downloads/ClassAccuracyMain_qgis3.zip](https://bitbucket.org/petebunting/classaccuracy/downloads/ClassAccuracyMain_qgis3.zip)

Within QGIS go to the 'Plugins' menu and select 'Manage and Install Plugins...'. (Figure 2)

<img src="figures/addplugin2qgis.png" alt="addplugin2qgis" style="zoom:60%;" />

Figure 2. Installing QGIS Plugin into QGIS 3.

You should then have a menu 'ClassAccuracy' within the 'Plugins' menu and a new icon on the toolbars.

### Using Existing Points

If you do not want to step through all the accuracy points in QGIS (it is highly recommended that you do this!!) then a set of accuracy points have been provided:

```bash
../dataset/acc_points/cls_acc_assessment_pts_ref.dbf
../dataset/acc_points/cls_acc_assessment_pts_ref.prj
../dataset/acc_points/cls_acc_assessment_pts_ref.shp
../dataset/acc_points/cls_acc_assessment_pts_ref.shx
```

These will need renaming so they can be used in place of the points generated above. Renaming the files is done with the following commands:

```bash
mv ../dataset/acc_points/cls_acc_assessment_pts_ref.dbf ../dataset/acc_points/cls_acc_assessment_pts.dbf
mv ../dataset/acc_points/cls_acc_assessment_pts_ref.prj ../dataset/acc_points/cls_acc_assessment_pts.prj
mv ../dataset/acc_points/cls_acc_assessment_pts_ref.shp ../dataset/acc_points/cls_acc_assessment_pts.shp
mv ../dataset/acc_points/cls_acc_assessment_pts_ref.shx ../dataset/acc_points/cls_acc_assessment_pts.shx
```

To populate the `cls_rf_refl.kea` classification on to these points run the following script, this will put you in the same position as if you had undertaken the accuracy assessment properly using the QGIS plugin assigning the reference points to their correct class.

```bash
python 04_setup_existing_accpts.py
```

### Calculating the Classification Accuracy

Once you have confirmed the class of all the reference points you can produce an error matrix using the plugin but a more complete option is to use the `rsgislib.classaccuracymetrics` function `calc_acc_metrics_vecsamples` which returns a number of accuracy metrics and estimation of the area of each class for the scene.

These can be calculated for the by running the script `06a_calc_class_acc_stats.py`, the code of which is shown below:

```bash
python 05_calc_class_acc_stats.py
```

This script is calculating the accuracy metrics for the random forests classifier using the original reflectance data.

```python
from classaccuracymetrics import calc_acc_metrics_vecsamples

calc_acc_metrics_vecsamples('../dataset/acc_points/cls_acc_assessment_pts.shp', 
                            'cls_acc_assessment_pts', 'ref_pts', 'rf_rl_cls', 
                            '../dataset/cls_results/cls_rf_refl.kea', 
                            out_csv_file='../dataset/cls_results/cls_rf_refl_acc_info.csv')

```

If you open the outputted CSV file (`cls_rf_refl_acc_info.csv`) the overall statistics which you should be paying attention to are:

- **Overall accuracy** - Based on the number of samples, this is the proportion of the those which are correct. If the number of sampled per class is equal (i.e., stratified by class) then this is telling you the ability of your classifier to seperate between the different classes.
- **Cohen kappa** - Also based on the number of samples (see above). An overall accuracy statistic which aims to remove the 'change agreement', i.e., the change of randomly guessing the correct class. While this measure has been traditionally used some authors have proposed that this measure should be replaced with the allocation and quantity disagreement (see below).
- **Proportion Correct** - This is the overall accuracy where the samples are normalised by the area of the class in the map and the number of samples per class. *This provides an overall accuracy of the map and should be the map accuracy quoted*.
- **Total Disagreement** - This is (1 - Proportion Correct) and is the total amount of the scene which has been incorrectly classified.
- **Allocation Disagreement** - The proportion of the disagreement (i.e., error) associated with pixels which are in the wrong spatial location.
- **Quantity Disagreement** - The proportion of the disagreement (i.e., error) associated with the amount classified

To interpret the allocation and quantity disagreement:

- If the allocation disagreement is high but the quantity disagreement is low then this is telling you that the area (amounts) classified is correct but the locations classified are incorrect.
- If the allocation disagreement is low but the quantity disagreement is high then this is telling you that the locations classified are correct but the area (amounts) classified are incorrect.

The other information outputted includes:

- The confusion matrix using the sample counts.
- The confusion matrix normalised by the number of samples and areas classified.
- For each class you have:
  - **f1-score** - The F1 score can be interpreted as a weighted average of the precision and recall, where an F1 score reaches its best value at 1 and worst score at 0. The relative contribution of precision and recall to the F1 score are equal.
  - **precision** - The precision is the ratio `tp / (tp + fp)` where `tp` is the number of true positives and `fp` the number of false positives. The precision is intuitively the ability of the classifier not to label as positive a sample that is negative.
  - **recall** - The recall is the ratio `tp / (tp + fn)` where `tp` is the number of true positives and `fn` the number of false negatives. The recall is intuitively the ability of the classifier to find all the positive samples.
  - **support** - number of samples used for the calculation
  - **commission** - this is the amount of overestimation for the class
  - **ommission** - this is the amount of underestimation for the class
  - **pixel count**  - the number of pixels classified in the map produced.
  - **area mapped** - the area classified in the map produced.
  - **proprotion of map** - an estimate, corrected by the accuracy assessment, of the proportion of the map classified as a particular class.
  - **corrected area mapped** - using the estimated proportion of the map to estimate the 'true' areas of each class within the scene.

### Compare Multiple Classifications

If you have a number of classifications to compare, as in this case, you will want to add columns for those classification to your shapefile. This can be done using the code below, which shows two input classifications:

```python
import rsgislib.classification

acc_points_shp = '../dataset/acc_points/cls_acc_assessment_pts.shp'

in_cls_img = '../dataset/cls_results/cls_ert_norm_sd.kea'
rsgislib.classification.popClassInfoAccuracyPts(in_cls_img, 
                                                acc_points_shp, 
                                                'ClassName', 
                                                'rf_nsd', 
                                                'ref0')

in_cls_img = '../dataset/cls_results/cls_ert_norm_lin.kea' 
rsgislib.classification.popClassInfoAccuracyPts(in_cls_img, 
                                                acc_points_shp, 
                                                'ClassName', 
                                                'rf_nln', 
                                                'ref1')

```

Running the following command will add all of the classification to the reference points:

```bash
python 06_add_cls_columns.py
```

You can now calculate the accuracy metrics for each of the classification using the same reference points, note that this also exports the statistics as a JSON file, which is easily machine readable so the statistics can be read and summarised by a follow on script. 

The following code calculates the error metrics for all the classifications:

```python
from classaccuracymetrics import calc_acc_metrics_vecsamples
import rsgislib

# The dict key is the column name and the value is the input image.
cls_inputs = dict()

# Random Forests
cls_inputs['rf_rl_cls'] = '../dataset/cls_results/cls_rf_refl.kea'
cls_inputs['rf_nln'] = '../dataset/cls_results/cls_rf_norm_lin.kea'
cls_inputs['rf_nsd'] = '../dataset/cls_results/cls_rf_norm_sd.kea'
# Maximum Likelihood
cls_inputs['ml_rl'] = '../dataset/cls_results/cls_ml_refl.kea'
cls_inputs['ml_nln'] = '../dataset/cls_results/cls_ml_norm_lin.kea'
cls_inputs['ml_nsd'] = '../dataset/cls_results/cls_ml_norm_sd.kea'
# Support Vector Machines
cls_inputs['svm_rl'] = '../dataset/cls_results/cls_svm_refl.kea'
cls_inputs['svm_nln'] = '../dataset/cls_results/cls_svm_norm_lin.kea'
cls_inputs['svm_nsd'] = '../dataset/cls_results/cls_svm_norm_sd.kea'
# K- Nearest Neighbour
cls_inputs['knn_rl'] = '../dataset/cls_results/cls_knn_refl.kea'
cls_inputs['knn_nln'] = '../dataset/cls_results/cls_knn_norm_lin.kea'
cls_inputs['knn_nsd'] = '../dataset/cls_results/cls_knn_norm_sd.kea'
# Extra Randomised Trees
cls_inputs['ert_rl'] = '../dataset/cls_results/cls_ert_refl.kea'
cls_inputs['ert_nln'] = '../dataset/cls_results/cls_ert_norm_lin.kea'
cls_inputs['ert_nsd'] = '../dataset/cls_results/cls_ert_norm_sd.kea'
# Gradient Boosted Decision Trees
cls_inputs['gbdt_rl'] = '../dataset/cls_results/cls_gbdt_refl.kea'
cls_inputs['gbdt_nln'] = '../dataset/cls_results/cls_gbdt_norm_lin.kea'
cls_inputs['gbdt_nsd'] = '../dataset/cls_results/cls_gbdt_norm_sd.kea'
# Neural Network
cls_inputs['nn_rl'] = '../dataset/cls_results/cls_nn_refl.kea'
cls_inputs['nn_nln'] = '../dataset/cls_results/cls_nn_norm_lin.kea'
cls_inputs['nn_nsd'] = '../dataset/cls_results/cls_nn_norm_sd.kea'

# Create instance of the rsgis utils class
rsgis_utils = rsgislib.RSGISPyUtils()

# Iterate through all the classifications and calculate the metrics
for cls_col in cls_inputs:
    basename = rsgis_utils.get_file_basename(cls_inputs[cls_col])
    calc_acc_metrics_vecsamples('../dataset/acc_points/cls_acc_assessment_pts.shp', 
                                'cls_acc_assessment_pts', 'ref_pts', cls_col, 
                                cls_inputs[cls_col], 
                                out_json_file='../dataset/cls_results/{}_acc_info.json'
                                                                     .format(basename),
                                out_csv_file='../dataset/cls_results/{}_acc_info.csv'
                                                                   .format(basename))

```

This can be executed using the following command:

```bash
python 07_calc_class_acc_stats.py
```

To summarise the accuracy metrics the script below reads in the input JSON files and creates a summary CSV file:

```python
import csv
import json

# Input list of classifications
# Key is class short reference
# Value is the accuracy metric statistics (JSON)
cls_inputs = dict()
# Random Forests
cls_inputs['rf_rl_cls'] = '../dataset/cls_results/cls_rf_refl_acc_info.json'
cls_inputs['rf_nln'] = '../dataset/cls_results/cls_rf_norm_lin_acc_info.json'
cls_inputs['rf_nsd'] = '../dataset/cls_results/cls_rf_norm_sd_acc_info.json'
# Maximum Likelihood
cls_inputs['ml_rl'] = '../dataset/cls_results/cls_ml_refl_acc_info.json'
cls_inputs['ml_nln'] = '../dataset/cls_results/cls_ml_norm_lin_acc_info.json'
cls_inputs['ml_nsd'] = '../dataset/cls_results/cls_ml_norm_sd_acc_info.json'
# Support Vector Machines
cls_inputs['svm_rl'] = '../dataset/cls_results/cls_svm_refl_acc_info.json'
cls_inputs['svm_nln'] = '../dataset/cls_results/cls_svm_norm_lin_acc_info.json'
cls_inputs['svm_nsd'] = '../dataset/cls_results/cls_svm_norm_sd_acc_info.json'
# K- Nearest Neighbour
cls_inputs['knn_rl'] = '../dataset/cls_results/cls_knn_refl_acc_info.json'
cls_inputs['knn_nln'] = '../dataset/cls_results/cls_knn_norm_lin_acc_info.json'
cls_inputs['knn_nsd'] = '../dataset/cls_results/cls_knn_norm_sd_acc_info.json'
# Extra Randomised Trees
cls_inputs['ert_rl'] = '../dataset/cls_results/cls_ert_refl_acc_info.json'
cls_inputs['ert_nln'] = '../dataset/cls_results/cls_ert_norm_lin_acc_info.json'
cls_inputs['ert_nsd'] = '../dataset/cls_results/cls_ert_norm_sd_acc_info.json'
# Gradient Boosted Decision Trees
cls_inputs['gbdt_rl'] = '../dataset/cls_results/cls_gbdt_refl_acc_info.json'
cls_inputs['gbdt_nln'] = '../dataset/cls_results/cls_gbdt_norm_lin_acc_info.json'
cls_inputs['gbdt_nsd'] = '../dataset/cls_results/cls_gbdt_norm_sd_acc_info.json'
# Neural Network
cls_inputs['nn_rl'] = '../dataset/cls_results/cls_nn_refl_acc_info.json'
cls_inputs['nn_nln'] = '../dataset/cls_results/cls_nn_norm_lin_acc_info.json'
cls_inputs['nn_nsd'] = '../dataset/cls_results/cls_nn_norm_sd_acc_info.json'

# Output summary metrics file
out_summary_file = '../dataset/cls_results/summary_cls_comparison.csv'
# Open the output CSV file.
with open(out_summary_file, mode='w') as out_csv_file_obj:
    # create the CSV file writer.
    sum_metrics_writer = csv.writer(out_csv_file_obj, delimiter=',', 
                                    quotechar='"', quoting=csv.QUOTE_MINIMAL)
    # Write the column headings to the file.
    sum_metrics_writer.writerow(['Class', 'Proportion Correct', 
                                 'Allocation Disagreement', 'Quantity Disagreement', 
                                 'Overall Accuracy', 'Kappa'])
    # Loop through the classifications 
    for cls in cls_inputs:
        # Create row for each classification approach.
        row_data = [cls]
        # Open the  JSON file.
        with open(cls_inputs[cls]) as json_file:
            # loads the JSON file, represented as a Python dict.
            cls_acc_data = json.load(json_file)
            # Add accuracy metrics to list to be written to row.
            row_data.append(cls_acc_data['quantity_metrics']['Proportion Correct (C)'])
            row_data.append(cls_acc_data['quantity_metrics']['Allocation Disagreement (A)'])
            row_data.append(cls_acc_data['quantity_metrics']['Quantity Disagreement (Q)'])
            row_data.append(cls_acc_data['accuracy'])
            row_data.append(cls_acc_data['cohen_kappa'])
        # Write row to CSV file.
        sum_metrics_writer.writerow(row_data)

```

This can be run with the following command:

```bash
python 08_summarise_class_acc_stats.py
```

The outputted summary statistics will product a table such as the one below (sorted by the 'Proportion Correct'). From the table below we can observe the classification using the 'Maximum Likelihood' using the data normalised using the standard deviation approach has provided the best map. However, it is suprising that the 'Gradient Boosted Decision Trees' and other more 'advanced' classifiers have not produced a better results as these algorithms are commonly referenced as producing good results, better than algorithms such as maximum likelihood.

| Class     | Proportion Correct | Allocation Disagreement | Quantity Disagreement | Overall Accuracy | Kappa |
| --------- | ------------------ | ----------------------- | --------------------- | ---------------- | ----- |
| ml_nsd    | 0.822              | 0.083                   | 0.095                 | 0.705            | 0.668 |
| ml_nln    | 0.813              | 0.095                   | 0.092                 | 0.702            | 0.665 |
| ml_rl     | 0.813              | 0.095                   | 0.092                 | 0.702            | 0.665 |
| knn_nln   | 0.810              | 0.117                   | 0.074                 | 0.725            | 0.694 |
| nn_rl     | 0.801              | 0.106                   | 0.093                 | 0.575            | 0.529 |
| knn_rl    | 0.798              | 0.121                   | 0.080                 | 0.719            | 0.687 |
| ert_nln   | 0.796              | 0.138                   | 0.066                 | 0.752            | 0.724 |
| ert_rl    | 0.794              | 0.147                   | 0.059                 | 0.760            | 0.732 |
| rf_rl_cls | 0.788              | 0.155                   | 0.057                 | 0.747            | 0.718 |
| ert_nsd   | 0.787              | 0.148                   | 0.065                 | 0.749            | 0.721 |
| knn_nsd   | 0.771              | 0.157                   | 0.073                 | 0.748            | 0.719 |
| gbdt_nln  | 0.764              | 0.109                   | 0.127                 | 0.730            | 0.699 |
| gbdt_rl   | 0.761              | 0.118                   | 0.120                 | 0.722            | 0.690 |
| rf_nln    | 0.759              | 0.180                   | 0.061                 | 0.729            | 0.698 |
| svm_nsd   | 0.759              | 0.114                   | 0.127                 | 0.680            | 0.647 |
| gbdt_nsd  | 0.758              | 0.119                   | 0.122                 | 0.719            | 0.687 |
| svm_rl    | 0.755              | 0.160                   | 0.084                 | 0.694            | 0.662 |
| svm_nln   | 0.749              | 0.175                   | 0.076                 | 0.693            | 0.660 |
| rf_nsd    | 0.746              | 0.187                   | 0.067                 | 0.722            | 0.690 |
| nn_nsd    | 0.632              | 0.246                   | 0.122                 | 0.598            | 0.554 |
| nn_nln    | 0.630              | 0.289                   | 0.081                 | 0.617            | 0.576 |



## Exercise 3. Optimising the Classifier Parameters

You ran each of the classifiers with their default parameters. However, the classifiers have many parameters and these can have a significant effect on the results of the classification. The simplest approach is to optimise the classifiers parameters is a 'Grid Search'. A grid search runs all of the different parameter options within the ranges you have provided, returning the classifier which produces the best result. 

You can run the script below using the following command:

```bash
python 09_apply_grid_search_extra_trees.py
```

In this script (below) you will see the following dictionary has been defined:

```python
param_grid={'n_estimators':[10,20,50,100,250,500,1000], 'max_depth':[2,4,6,8]}
```

This dictionary is defining the values the grid search is going to test, i.e., all the possible combinations of these parameters. Clearly, the more variables and values, for that variable, you specify the long it will take to compute. Try replacing the dictionary with the one below and comparing how long it takes to run:

```python
param_grid={'n_estimators':[10,20,50,100,250,500,1000], 
            'max_depth':[2,4,6,8],
            'criterion':['gini','entropy'],
            'min_samples_split':[2,3,4],
            'max_features':['auto','sqrt','log2',None]}
```

It should have taken quite a bit longer to execute. In the LightGBM section we will use at an alterntive method (Baysian Optimisation) to optimise the parameters, which is potentially more efficient.

The full script to run a grid search optimised classificstion is given below.

```python
import os.path
from rsgislib.imageutils import ImageBandInfo
from rsgislib.classification import ClassSimpleInfoObj
from rsgislib.classification import classsklearn
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV

out_cls_dir = '../dataset/cls_gridsearch_results'
if not os.path.exists(out_cls_dir):
    raise Exception("Output Directory does not exist, please create.")

refl_img_file = '../dataset/SEN2_20180629_T30UVD_ORB037_osgb_stdsref.tif'
imgs_info = []
imgs_info.append(ImageBandInfo(fileName=refl_img_file, name='sen2', 
                               bands=[1,2,3,4,5,6,7,8,9,10]))

valid_img = '../dataset/SEN2_20180629_T30UVD_ORB037_osgb_stdsref_vmsk.kea'

cls_train_info = dict()
cls_train_info['Artificial_Surfaces'] = ClassSimpleInfoObj(id=1, 
            fileH5='../dataset/training/refl_train_samples_Artificial_Surfaces.h5', 
            red=160, green=160, blue=160)
cls_train_info['Bare_Rock_Sand']      = ClassSimpleInfoObj(id=2, 
            fileH5='../dataset/training/refl_train_samples_Bare_Rock_Sand.h5', 
            red=100, green=100, blue=100)
cls_train_info['Bracken']             = ClassSimpleInfoObj(id=3, 
            fileH5='../dataset/training/refl_train_samples_Bracken.h5', 
            red=235, green=146, blue=38)
cls_train_info['Conifer_Forest']      = ClassSimpleInfoObj(id=4, 
            fileH5='../dataset/training/refl_train_samples_Conifer_Forest.h5', 
            red=2, green=97, blue=16)
cls_train_info['Deciduous_Forest']    = ClassSimpleInfoObj(id=5, 
            fileH5='../dataset/training/refl_train_samples_Deciduous_Forest.h5', 
            red=50, green=184, blue=69)
cls_train_info['Grass_Long']          = ClassSimpleInfoObj(id=6, 
            fileH5='../dataset/training/refl_train_samples_Grass_Long.h5', 
            red=150, green=227, blue=18)
cls_train_info['Grass_Short']         = ClassSimpleInfoObj(id=7, 
            fileH5='../dataset/training/refl_train_samples_Grass_Short.h5',
            red=194, green=219, blue=66)
cls_train_info['NonPhoto_Veg']        = ClassSimpleInfoObj(id=8, 
            fileH5='../dataset/training/refl_train_samples_NonPhotosynthetic_Vegetation.h5', 
            red=98, green=225, blue=227)
cls_train_info['Scrub']               = ClassSimpleInfoObj(id=9, 
            fileH5='../dataset/training/refl_train_samples_Scrub.h5', 
            red=0, green=80, blue=0)
cls_train_info['Water']               = ClassSimpleInfoObj(id=10, 
            fileH5='../dataset/training/refl_train_samples_Water_Training.h5', 
            red=0, green=0, blue=255)

# Train the classifier
grid_search = GridSearchCV(ExtraTreesClassifier(bootstrap=True), 
                           param_grid={'n_estimators':[10,20,50,100,250,500,1000], 
                                       'max_depth':[2,4,6,8]})
skclf = classsklearn.train_sklearn_classifer_gridsearch(cls_train_info, 500, grid_search)

# Apply the classifier to the image
out_cls_img = os.path.join(out_cls_dir, 'cls_ertgs_refl.kea')
classsklearn.apply_sklearn_classifer(cls_train_info, skclf, valid_img, 1, imgs_info, 
                                     out_cls_img, 'KEA', classClrNames=True)

```

After running the 'best' classifier parameters are found:

```bash
Best score was 0.9702 and has parameters {'max_depth': 8, 'n_estimators': 50}.
```

If you run the analysis again are the same 'optimal' parameters found? If they change each time you run the algorithm what does this tell you about the optimisation? (i.e., it isn't stable)

Does this produce a better classification than the previous classifiers you ran? 

You can run the following script to calculate the accuracy of the map produced using this parameter optimised classifier.

```bash
python 10_grid_search_acc_metrics.py
```

The code within `10_grid_search_acc_metrics.py` is shown below:

```python
from classaccuracymetrics import calc_acc_metrics_vecsamples
import rsgislib.classification

acc_points_shp = '../dataset/acc_points/cls_acc_assessment_pts.shp'
in_cls_img = '../dataset/cls_gridsearch_results/cls_rf_refl.kea'                                            
rsgislib.classification.popClassInfoAccuracyPts(in_cls_img, 
                                                acc_points_shp, 
                                                'ClassName', 
                                                'rf_gs_rl', 
                                                'ref20')

out_json_file='../dataset/cls_gridsearch_results/rf_gs_rl_acc_info.json'
out_csv_file='../dataset/cls_gridsearch_results/rf_gs_rl_acc_info.csv'

calc_acc_metrics_vecsamples(acc_points_shp, 'cls_acc_assessment_pts', 
                            'ref_pts', 'rf_gs_rl', in_cls_img, 
                            out_json_file=out_json_file,
                            out_csv_file=out_csv_file)
```



## Exercise 4. Apply a LightGBM and Keras Classification

Beyond the scikit-learn library there are other classification approaches and libraries available to Python, such as the [LightGBM](https://lightgbm.readthedocs.io) and [Keras](https://keras.io) libraries, but also others such as [H20.ia](http://docs.h2o.ai), [XGBoost](https://xgboost.readthedocs.io), [CatBoost](https://catboost.ai) and [PyTorch](https://pytorch.org) are also available, although functions to interface with this libraries are not yet available within RSGISLib.

### Classifying with LightGBM

The LightGBM classifier is an implementation of Gradient Boosted Decision Trees (GBDT) and has a very large number of parameters, and these need to optimised. However, a grid search would be too computational complex undertaken, taking a very long time to complete. Therefore a Bayesian Optimisation can be used to find the classifier parameters. Bayesian Optimisation is an approach that uses Bayes Theorem to direct the search in order to find the minimum or maximum of an unknown objective function. The Baysian Optimisation will be undertaken within the RSGISLib LightGBM training function using the scikit-optimize (https://scikit-optimize.github.io) library.

The script (`11_apply_lightgbm_class.py`) to run a LightGBM classification uses the same training data as the earlier classifiers and is given below:

```python
import os.path
from rsgislib.imageutils import ImageBandInfo
from rsgislib.classification import ClassInfoObj
import classlightgbm

out_cls_dir = '../dataset/cls_lightgbm'
if not os.path.exists(out_cls_dir):
    raise Exception("Output Directory does not exist, please create.")

valid_img = '../dataset/SEN2_20180629_T30UVD_ORB037_osgb_stdsref_vmsk.kea'

refl_img_file = '../dataset/SEN2_20180629_T30UVD_ORB037_osgb_stdsref.tif'
imgs_info = []
imgs_info.append(ImageBandInfo(fileName=refl_img_file, name='sen2', 
                               bands=[1,2,3,4,5,6,7,8,9,10]))

cls_train_info = dict()
cls_train_info['Artificial_Surfaces'] = ClassInfoObj(id=0, out_id=1, 
        trainfileH5="../dataset/training/refl_train_samples_Artificial_Surfaces.h5", 
        testfileH5="../dataset/training/refl_test_samples_Artificial_Surfaces.h5", 
        validfileH5="../dataset/training/refl_valid_samples_Artificial_Surfaces.h5", 
        red=160, green=160, blue=160)
cls_train_info['Bare_Rock_Sand']      = ClassInfoObj(id=1, out_id=2, 
        trainfileH5="../dataset/training/refl_train_samples_Bare_Rock_Sand.h5", 
        testfileH5="../dataset/training/refl_test_samples_Bare_Rock_Sand.h5", 
        validfileH5="../dataset/training/refl_valid_samples_Bare_Rock_Sand.h5", 
        red=100, green=100, blue=100)
cls_train_info['Bracken']             = ClassInfoObj(id=2, out_id=3, 
        trainfileH5="../dataset/training/refl_train_samples_Bracken.h5", 
        testfileH5="../dataset/training/refl_test_samples_Bracken.h5", 
        validfileH5="../dataset/training/refl_valid_samples_Bracken.h5",
        red=235, green=146, blue=38)
cls_train_info['Conifer_Forest']      = ClassInfoObj(id=3, out_id=4, 
        trainfileH5="../dataset/training/refl_train_samples_Conifer_Forest.h5", 
        testfileH5="../dataset/training/refl_test_samples_Conifer_Forest.h5", 
        validfileH5="../dataset/training/refl_valid_samples_Conifer_Forest.h5", 
        red=2, green=97, blue=16)
cls_train_info['Deciduous_Forest']    = ClassInfoObj(id=4, out_id=5, 
        trainfileH5="../dataset/training/refl_train_samples_Deciduous_Forest.h5",
        testfileH5="../dataset/training/refl_test_samples_Deciduous_Forest.h5",
        validfileH5="../dataset/training/refl_valid_samples_Deciduous_Forest.h5",
        red=50, green=184, blue=69)
cls_train_info['Grass_Long']          = ClassInfoObj(id=5, out_id=6, 
        trainfileH5="../dataset/training/refl_train_samples_Grass_Long.h5", 
        testfileH5="../dataset/training/refl_test_samples_Grass_Long.h5", 
        validfileH5="../dataset/training/refl_valid_samples_Grass_Long.h5", 
        red=150, green=227, blue=18)
cls_train_info['Grass_Short']         = ClassInfoObj(id=6, out_id=7, 
        trainfileH5="../dataset/training/refl_train_samples_Grass_Short.h5", 
        testfileH5="../dataset/training/refl_test_samples_Grass_Short.h5", 
        validfileH5="../dataset/training/refl_valid_samples_Grass_Short.h5", 
        red=194, green=219, blue=66)
cls_train_info['NonPhoto_Veg']        = ClassInfoObj(id=7, out_id=8, 
        trainfileH5="../dataset/training/refl_train_samples_NonPhotosynthetic_Vegetation.h5", 
        testfileH5="../dataset/training/refl_test_samples_NonPhotosynthetic_Vegetation.h5", 
        validfileH5="../dataset/training/refl_valid_samples_NonPhotosynthetic_Vegetation.h5", 
        red=98, green=225, blue=227)
cls_train_info['Scrub']               = ClassInfoObj(id=8, out_id=9, 
        trainfileH5="../dataset/training/refl_train_samples_Scrub.h5", 
        testfileH5="../dataset/training/refl_test_samples_Scrub.h5", 
        validfileH5="../dataset/training/refl_valid_samples_Scrub.h5", 
        red=0, green=80, blue=0)
cls_train_info['Water']               = ClassInfoObj(id=9, out_id=10, 
        trainfileH5="../dataset/training/refl_train_samples_Water_Training.h5", 
        testfileH5="../dataset/training/refl_test_samples_Water_Training.h5", 
        validfileH5="../dataset/training/refl_valid_samples_Water_Training.h5", 
        red=0, green=0, blue=255)

# Train the classifier
out_mdl_file = os.path.join(out_cls_dir, 'refl_cls_lgbm_mdl.txt')
classlightgbm.train_lightgbm_multiclass_classifer(out_mdl_file, cls_train_info)

# Apply the classifier to the image
out_cls_img = os.path.join(out_cls_dir, 'cls_lgbm_refl.kea')
out_cls_prob_img = os.path.join(out_cls_dir, 'cls_lgbm_refl_prob.kea')
classlightgbm.apply_lightgbm_multiclass_classifier(cls_train_info, out_mdl_file, 
                                                   valid_img, 1, imgs_info, 
                                                   out_cls_prob_img, out_cls_img, 'KEA')

```

To run this script using the command below:

```bash
python 11_apply_lightgbm_class.py
```

Remember to create the output directory:

```bash
mkdir ../dataset/cls_lightgbm
```

The LightGBM training can take sometime and the model created is saved as a text file (`*.txt`) so you can then use this model to classify other scenes without re-training the model.

#### Accuracy of the LightGBM Classification

Using the same reference points, developed earlier, we can assess the accuracy fo the classification:

```bash
python 12_calc_lightgbm_class_acc_metrics.py
```

The code of which will be familiar to you, and show below:

```python
from classaccuracymetrics import calc_acc_metrics_vecsamples
import rsgislib.classification

acc_points_shp = '../dataset/acc_points/cls_acc_assessment_pts.shp'
in_cls_img = '../dataset/cls_lightgbm/cls_lgbm_refl.kea'                                            rsgislib.classification.popClassInfoAccuracyPts(in_cls_img, 
                                                acc_points_shp, 
                                                'ClassName', 
                                                'lgbm_rl', 
                                                'ref21')

out_json_file='../dataset/cls_lightgbm/lgbm_rl_acc_info.json'
out_csv_file='../dataset/cls_lightgbm/lgbm_rl_acc_info.csv'

calc_acc_metrics_vecsamples(acc_points_shp, 'cls_acc_assessment_pts', 
                            'ref_pts', 'lgbm_rl', in_cls_img, 
                            out_json_file=out_json_file,
                            out_csv_file=out_csv_file)

```



### Classifying with Keras

The next exercise uses the Keras library which allows for complex neural network classifiers to be build and applied. These large networks are commonly referred to as 'deep learning' classifiers. The script structure is similar to those above but you must define the network structure for the classifier, shown below:

```python
cls_mdl = Sequential()
# The input_dim must be the same as the number of image 
# bands used for the classification (i.e., 10)
cls_mdl.add(Dense(32, activation='relu', input_dim=10))
cls_mdl.add(Dense(16, activation='relu'))
cls_mdl.add(Dense(8,  activation='relu'))
cls_mdl.add(Dense(32, activation='relu'))
# The final layer of the network must use softmax activation and 
# the size must be the same as the number of classes (i.e., 10)
cls_mdl.add(Dense(10, activation='softmax'))
cls_mdl.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
```

The structure has 5 layers:

1. 10 input dimensions, which relates to the number of image bands being used for the classification. This connects to 32 neurons. 
2. 16 neuron layer
3. 8 neuron layer
4. 32 neuron layer
5. 10 neuron layer which relates to the number of output classes.

The full script is shown below:

```python
import os.path
from rsgislib.imageutils import ImageBandInfo
from rsgislib.classification import ClassInfoObj
import classkeraspxl
from keras.models import Sequential
from keras.layers import Dense, Activation

out_cls_dir = '../dataset/cls_keras_pxl'
if not os.path.exists(out_cls_dir):
    raise Exception("Output Directory does not exist, please create.")

valid_img = '../dataset/SEN2_20180629_T30UVD_ORB037_osgb_stdsref_vmsk.kea'

refl_img_file = '../dataset/SEN2_20180629_T30UVD_ORB037_osgb_stdsref.tif'
imgs_info = []
imgs_info.append(ImageBandInfo(fileName=refl_img_file, name='sen2', 
                               bands=[1,2,3,4,5,6,7,8,9,10]))

cls_train_info = dict()
cls_train_info['Artificial_Surfaces'] = ClassInfoObj(id=0, out_id=1, 
        trainfileH5="../dataset/training/refl_train_samples_Artificial_Surfaces.h5", 
        testfileH5="../dataset/training/refl_test_samples_Artificial_Surfaces.h5", 
        validfileH5="../dataset/training/refl_valid_samples_Artificial_Surfaces.h5", 
        red=160, green=160, blue=160)
cls_train_info['Bare_Rock_Sand']      = ClassInfoObj(id=1, out_id=2, 
        trainfileH5="../dataset/training/refl_train_samples_Bare_Rock_Sand.h5", 
        testfileH5="../dataset/training/refl_test_samples_Bare_Rock_Sand.h5", 
        validfileH5="../dataset/training/refl_valid_samples_Bare_Rock_Sand.h5", 
        red=100, green=100, blue=100)
cls_train_info['Bracken']             = ClassInfoObj(id=2, out_id=3, 
        trainfileH5="../dataset/training/refl_train_samples_Bracken.h5", 
        testfileH5="../dataset/training/refl_test_samples_Bracken.h5", 
        validfileH5="../dataset/training/refl_valid_samples_Bracken.h5",
        red=235, green=146, blue=38)
cls_train_info['Conifer_Forest']      = ClassInfoObj(id=3, out_id=4, 
        trainfileH5="../dataset/training/refl_train_samples_Conifer_Forest.h5", 
        testfileH5="../dataset/training/refl_test_samples_Conifer_Forest.h5", 
        validfileH5="../dataset/training/refl_valid_samples_Conifer_Forest.h5", 
        red=2, green=97, blue=16)
cls_train_info['Deciduous_Forest']    = ClassInfoObj(id=4, out_id=5, 
        trainfileH5="../dataset/training/refl_train_samples_Deciduous_Forest.h5",
        testfileH5="../dataset/training/refl_test_samples_Deciduous_Forest.h5",
        validfileH5="../dataset/training/refl_valid_samples_Deciduous_Forest.h5",
        red=50, green=184, blue=69)
cls_train_info['Grass_Long']          = ClassInfoObj(id=5, out_id=6, 
        trainfileH5="../dataset/training/refl_train_samples_Grass_Long.h5", 
        testfileH5="../dataset/training/refl_test_samples_Grass_Long.h5", 
        validfileH5="../dataset/training/refl_valid_samples_Grass_Long.h5", 
        red=150, green=227, blue=18)
cls_train_info['Grass_Short']         = ClassInfoObj(id=6, out_id=7, 
        trainfileH5="../dataset/training/refl_train_samples_Grass_Short.h5", 
        testfileH5="../dataset/training/refl_test_samples_Grass_Short.h5", 
        validfileH5="../dataset/training/refl_valid_samples_Grass_Short.h5", 
        red=194, green=219, blue=66)
cls_train_info['NonPhoto_Veg']        = ClassInfoObj(id=7, out_id=8, 
        trainfileH5="../dataset/training/refl_train_samples_NonPhotosynthetic_Vegetation.h5", 
        testfileH5="../dataset/training/refl_test_samples_NonPhotosynthetic_Vegetation.h5", 
        validfileH5="../dataset/training/refl_valid_samples_NonPhotosynthetic_Vegetation.h5", 
        red=98, green=225, blue=227)
cls_train_info['Scrub']               = ClassInfoObj(id=8, out_id=9, 
        trainfileH5="../dataset/training/refl_train_samples_Scrub.h5", 
        testfileH5="../dataset/training/refl_test_samples_Scrub.h5", 
        validfileH5="../dataset/training/refl_valid_samples_Scrub.h5", 
        red=0, green=80, blue=0)
cls_train_info['Water']               = ClassInfoObj(id=9, out_id=10, 
        trainfileH5="../dataset/training/refl_train_samples_Water_Training.h5", 
        testfileH5="../dataset/training/refl_test_samples_Water_Training.h5", 
        validfileH5="../dataset/training/refl_valid_samples_Water_Training.h5", 
        red=0, green=0, blue=255)

cls_mdl = Sequential()
# The input_dim must be the same as the number of image 
# bands used for the classification (i.e., 10)
cls_mdl.add(Dense(32, activation='relu', input_dim=10))
cls_mdl.add(Dense(16, activation='relu'))
cls_mdl.add(Dense(8,  activation='relu'))
cls_mdl.add(Dense(32, activation='relu'))
# The final layer of the network must use softmax activation and 
# the size must be the same as the number of classes (i.e., 10)
cls_mdl.add(Dense(10, activation='softmax'))
cls_mdl.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the classifier
out_mdl_file = os.path.join(out_cls_dir, 'refl_cls_keras_mdl.h5')
classkeraspxl.train_keras_pixel_classifer(cls_mdl, cls_train_info, 
                                          out_mdl_file, train_epochs=20, 
                                          train_batch_size=32)

# Apply the classifier to the image
out_cls_img = os.path.join(out_cls_dir, 'cls_keras_refl.kea')
classkeraspxl.apply_keras_pixel_classifier(cls_train_info, cls_mdl, 
                                           valid_img, 1, imgs_info, 
                                           out_cls_img, 'KEA')

```

This is executed using the following command:

```bash
python 13_apply_keras_class.py
```

#### Accuracy of the Keras Pixel Classification

Using the same reference points, developed earlier, we can assess the accuracy fo the classification:

```bash
python 14_calc_keras_class_acc_metrics.py
```

The code of which will be familiar to you, and show below:

```python
from classaccuracymetrics import calc_acc_metrics_vecsamples
import rsgislib.classification

acc_points_shp = '../dataset/acc_points/cls_acc_assessment_pts.shp'
in_cls_img = '../dataset/cls_keras_pxl/cls_keras_refl.kea'
rsgislib.classification.popClassInfoAccuracyPts(in_cls_img, 
                                                acc_points_shp, 
                                                'ClassName', 
                                                'keras_rl', 
                                                'ref22')

out_json_file='../dataset/cls_keras_pxl/keras_rl_acc_info.json'
out_csv_file='../dataset/cls_keras_pxl/keras_rl_acc_info.csv'

calc_acc_metrics_vecsamples(acc_points_shp, 'cls_acc_assessment_pts', 
                            'ref_pts', 'keras_rl', in_cls_img,
                            out_json_file=out_json_file,
                            out_csv_file=out_csv_file)

```

## Exercise 5. Creating a Hierarchical Classification

In the classifications so far undertake all the classes have been directly compared to one another. However, using the image mask input on all the classification functions you can select which pixels are being considered for the classification and therefore create a hierarchical. The structure used for the following example is shown in Figure below.

<img src="figures/heirarchical_classification_structure.png" alt="heirarchical_classification_structure" style="zoom:67%;" />

Note, for this analysis you will need to create a new output directory:

```bash
mkdir ../dataset/cls_hierachical
```

### Classify Photosynthetic Surfaces / Non-Photosynthetic Surfaces

To split the first level, photosynthetic / non-photoscynthetic surfaces you first need to merge the training data (`15a_apply_veg_nonveg_split_samples.py`) and then use this training data to classify the scene (`15b_apply_veg_nonveg_split_class.py`).

For the first step run the following script, the code of which is shown below:

```bash
python 15a_apply_veg_nonveg_split_samples.py
```

```python
import rsgislib.imageutils

# Train
rsgislib.imageutils.mergeExtractedHDF5Data(
         ['../dataset/training/refl_train_samples_Bracken.h5', 
          '../dataset/training/refl_train_samples_Conifer_Forest.h5',
          '../dataset/training/refl_train_samples_Deciduous_Forest.h5',
          '../dataset/training/refl_train_samples_Grass_Long.h5',
          '../dataset/training/refl_train_samples_Grass_Short.h5',
          '../dataset/training/refl_train_samples_Scrub.h5'], 
          '../dataset/cls_hierachical/veg_train_cls_samples.h5')

rsgislib.imageutils.mergeExtractedHDF5Data(
         ['../dataset/training/refl_train_samples_Artificial_Surfaces.h5', 
          '../dataset/training/refl_train_samples_Bare_Rock_Sand.h5',
          '../dataset/training/refl_train_samples_NonPhotosynthetic_Vegetation.h5',
          '../dataset/training/refl_train_samples_Water_Training.h5'], 
          '../dataset/cls_hierachical/nonveg_train_cls_samples.h5')

# Test
rsgislib.imageutils.mergeExtractedHDF5Data(
         ['../dataset/training/refl_test_samples_Bracken.h5', 
          '../dataset/training/refl_test_samples_Conifer_Forest.h5',
          '../dataset/training/refl_test_samples_Deciduous_Forest.h5',
          '../dataset/training/refl_test_samples_Grass_Long.h5',
          '../dataset/training/refl_test_samples_Grass_Short.h5',
          '../dataset/training/refl_test_samples_Scrub.h5'], 
          '../dataset/cls_hierachical/veg_test_cls_samples.h5')

rsgislib.imageutils.mergeExtractedHDF5Data(
         ['../dataset/training/refl_test_samples_Artificial_Surfaces.h5', 
          '../dataset/training/refl_test_samples_Bare_Rock_Sand.h5',
          '../dataset/training/refl_test_samples_NonPhotosynthetic_Vegetation.h5',
          '../dataset/training/refl_test_samples_Water_Training.h5'], 
          '../dataset/cls_hierachical/nonveg_test_cls_samples.h5')

# Valid
rsgislib.imageutils.mergeExtractedHDF5Data(
         ['../dataset/training/refl_valid_samples_Bracken.h5', 
          '../dataset/training/refl_valid_samples_Conifer_Forest.h5',
          '../dataset/training/refl_valid_samples_Deciduous_Forest.h5',
          '../dataset/training/refl_valid_samples_Grass_Long.h5',
          '../dataset/training/refl_valid_samples_Grass_Short.h5',
          '../dataset/training/refl_valid_samples_Scrub.h5'], 
          '../dataset/cls_hierachical/veg_valid_cls_samples.h5')

rsgislib.imageutils.mergeExtractedHDF5Data(
         ['../dataset/training/refl_valid_samples_Artificial_Surfaces.h5', 
          '../dataset/training/refl_valid_samples_Bare_Rock_Sand.h5',
          '../dataset/training/refl_valid_samples_NonPhotosynthetic_Vegetation.h5',
          '../dataset/training/refl_valid_samples_Water_Training.h5'], 
          '../dataset/cls_hierachical/nonveg_valid_cls_samples.h5')

```

To perform the classification, run the following script:

```bash
python 15b_apply_veg_nonveg_split_class.py
```

The code for this script is below:

```python
import rsgislib
import rsgislib.imagecalc 
from rsgislib.classification import classlightgbm
from rsgislib.imageutils import ImageBandInfo

refl_img_file = '../dataset/SEN2_20180629_T30UVD_ORB037_osgb_stdsref.tif'
imgs_info = []
imgs_info.append(ImageBandInfo(fileName=refl_img_file, name='sen2', 
                               bands=[1,2,3,4,5,6,7,8,9,10]))

valid_img = '../dataset/SEN2_20180629_T30UVD_ORB037_osgb_stdsref_vmsk.kea'

veg_train = '../dataset/cls_hierachical/veg_train_cls_samples.h5'
veg_test = '../dataset/cls_hierachical/veg_test_cls_samples.h5'
veg_valid = '../dataset/cls_hierachical/veg_valid_cls_samples.h5'

non_veg_train = '../dataset/cls_hierachical/nonveg_train_cls_samples.h5'
non_veg_test = '../dataset/cls_hierachical/veg_test_cls_samples.h5'
non_veg_valid = '../dataset/cls_hierachical/nonveg_valid_cls_samples.h5'

out_mdl_file = '../dataset/cls_hierachical/veg_nonveg_lgbm_mdl.txt'
classlightgbm.train_lightgbm_binary_classifer(out_mdl_file, veg_train, 
                                              veg_valid, veg_test, 
                                              non_veg_train, non_veg_valid, 
                                              non_veg_test,
                                               unbalanced=True)

out_prob_img = '../dataset/cls_hierachical/veg_prob_cls.kea'
out_cls_img = '../dataset/cls_hierachical/veg_cls.kea'
classlightgbm.apply_lightgbm_binary_classifier(out_mdl_file, valid_img, 
                                               1, imgs_info, out_prob_img, 
                                               'KEA', out_cls_img, 
                                               class_thres=5000)

bandDefns = []
bandDefns.append(rsgislib.imagecalc.BandDefn('veg', out_cls_img, 1))
bandDefns.append(rsgislib.imagecalc.BandDefn('msk', valid_img, 1))
out_cls_fnl_img = '../dataset/cls_hierachical/veg_nonveg_cls.kea'
rsgislib.imagecalc.bandMath(out_cls_fnl_img, 
                   '(msk==1)&&(veg==1)?1:(msk==1)&&(veg==0)?2:0', 
                   'KEA', rsgislib.TYPE_8UINT, bandDefns)

rsgislib.rastergis.populateStats(out_cls_fnl_img, addclrtab=True, 
                                 calcpyramids=True, ignorezero=True)

```

### Classify Non Photosynthetic Surfaces

To merge the samples for the non-photosynthetic surfaces, use the following script:

```bash
python 16a_apply_nonveg_samples.py
```

To perform the classification, run the following script:

```bash
python 16b_apply_nonveg_class.py
```

Note, the classifications parts are merged using `bandMath`.

```python
bandDefns = []
bandDefns.append(rsgislib.imagecalc.BandDefn('bare', out_bare_cls_img, 1))
bandDefns.append(rsgislib.imagecalc.BandDefn('nonveg', out_nonveg_cls_img, 1))
out_cls_fnl_img = '../dataset/cls_hierachical/nonveg_fnl_cls.kea'
rsgislib.imagecalc.bandMath(out_cls_fnl_img, 
                   '(bare==1)?1:(bare==2)?2:(nonveg==2)?3:(nonveg==3)?4:0', 
                   'KEA', rsgislib.TYPE_8UINT, bandDefns)

rsgislib.rastergis.populateStats(out_cls_fnl_img, addclrtab=True, 
                                 calcpyramids=True, ignorezero=True)

```

### Classify Woody Surfaces

The woody surfaces are classified next, separating them from the other photosynthetic surfaces, this can be done by executing the following script:

```bash
python 17_apply_photoveg_woody_class.py
```

To classify the woody vegetation type, the following scripts  is to be used:

```bash
python 18_apply_woody_class.py
```

### Classify Other Vegetation Surfaces

To classify the other vegetation surfaces, use the following script

```bash
python 19_apply_otherveg_class.py
```

### Merge the Classifications

To merge the classifications you can use bandMaths, script:

```bash
python 20_merge_cls_fnl_cls.py
```

```python
import os
import rsgislib
import rsgislib.imagecalc
import rsgislib.rastergis
import osgeo.gdal as gdal
from rios import rat
import numpy

out_cls_dir = '../dataset/cls_hierachical'
if not os.path.exists(out_cls_dir):
    raise Exception("Output Directory does not exist, please create.")

non_veg_cls = '../dataset/cls_hierachical/nonveg_fnl_cls.kea'
woody_veg_cls = '../dataset/cls_hierachical/woody_types_cls_lgbm.kea'
nonwoody_veg_cls = '../dataset/cls_hierachical/otherveg_types_cls_lgbm.kea'

bandDefns = []
bandDefns.append(rsgislib.imagecalc.BandDefn('nonveg', non_veg_cls, 1))
bandDefns.append(rsgislib.imagecalc.BandDefn('woody', woody_veg_cls, 1))
bandDefns.append(rsgislib.imagecalc.BandDefn('oveg', nonwoody_veg_cls, 1))
out_cls_fnl_img = '../dataset/cls_hierachical/fnl_hier_cls.kea'
exp = """(nonveg==1)?1:(nonveg==2)?2:(nonveg==3)?8:
         (nonveg==4)?10:(woody==1)?4:(woody==2)?5:
         (oveg==1)?3:(oveg==2)?6:(oveg==3)?7:(oveg==4)?9:0"""
rsgislib.imagecalc.bandMath(out_cls_fnl_img, 
                   exp, 
                   'KEA', rsgislib.TYPE_8UINT, bandDefns)

rsgislib.rastergis.populateStats(out_cls_fnl_img, addclrtab=True, 
                                 calcpyramids=True, ignorezero=True)

ratDataset = gdal.Open(out_cls_fnl_img, gdal.GA_Update)
red = rat.readColumn(ratDataset, 'Red')
green = rat.readColumn(ratDataset, 'Green')
blue = rat.readColumn(ratDataset, 'Blue')
ClassName = numpy.empty_like(red, dtype=numpy.dtype('a255'))
ClassName[...] = ""

red[1]       = 160
green[1]     = 160
blue[1]      = 160
ClassName[1] = 'Artificial_Surfaces'

red[2]       = 100
green[2]     = 100
blue[2]      = 100
ClassName[2] = 'Bare_Rock_Sand'

red[3]       = 235
green[3]     = 146
blue[3]      = 38
ClassName[3] = 'Bracken'

red[4]       = 2
green[4]     = 97
blue[4]      = 16
ClassName[4] = 'Conifer_Forest'

red[5]       = 50
green[5]     = 184
blue[5]      = 69
ClassName[5] = 'Deciduous_Forest'

red[6]       = 150
green[6]     = 227
blue[6]      = 18
ClassName[6] = 'Grass_Long'

red[7]       = 194
green[7]     = 219
blue[7]      = 66
ClassName[7] = 'Grass_Short'

red[8]       = 98
green[8]     = 225
blue[8]      = 227
ClassName[8] = 'NonPhoto_Veg'

red[9]       = 0
green[9]     = 80
blue[9]      = 0
ClassName[9] = 'Scrub'

red[10]       = 0
green[10]     = 0
blue[10]      = 255
ClassName[10] = 'Water'

rat.writeColumn(ratDataset, "Red", red)
rat.writeColumn(ratDataset, "Green", green)
rat.writeColumn(ratDataset, "Blue", blue)
rat.writeColumn(ratDataset, "ClassName", ClassName)
ratDataset = None

```

### Accuracy Assessment

As with the other classifications, an accuracy assessment is required and can be perfomed using the following script:

```bash
python 21_cls_heir_accuracy_metrics.py
```

## Summary

You should now be able to run these classifiers through imagery of your own and solve your own classification problems. Things to have a think about:

1. Which is the 'best' classifier for your problem? 
2. How can you identify out which is the best classifier for your problem?
3. How can you optimise the parameters for the classifier you are using and ensure those 'really' are the 'best' parameters?
4. How much training data do you need? 
5. Where in the scene should training data be taken? Be careful of spatial autocorrelation (i.e., site colated with many pixels next to one other rather than independent.) 
6. How transferable is your classifier? Can you apply your classifier to other Sentinel-2 images?