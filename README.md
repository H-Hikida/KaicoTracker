# KaicoTracker
KaicoTracker was developed for tracking locomotion of silkworm larvae. The software consists three parts:

+ Detection of silkworm larvae and its position
+ Analysing thier locomotion
+ Validation the tracking results

Each part can be run separately, and data created from other process can be analysed if its format is consistent. KaicoTracker is developed for analyzing the locomotion of each individual, which are separately placed. For example, each larva was separated in cell culture dish in our test data.

---
### Dependencies
KaicoTracker is written in Python 3.6.3 with pandas (1.1.5), scipy (1.5.4), matplotlib (3.0.2), seaborn (0.11.1), and OpenCV2 (4.5.2.54). Please note that OpenCV version is 2.

---
### Installation
To start

> git clone https://github.com/H-Hikida/KaicoTracker  
>  
> cd KaicoTracker  
>  
> python KaicoTracker.py -h  

---
### Basic usage
Test data is provided in the repository, so that users can test if the program woks or not. It is recommended to use `--auroCrop` option with the information of how larvae were separated by Columns/Rows. It prevent to create some useless data and noises.

> KaicoTracker.py --input vtest_color_200.avi --autoCrop 4 5

The regular output will be:

##### Detection part
*_XM_kde.png

*_YM_kde.png

*.txt

##### Analyzing part
*_positions.txt

*_dist_data.txt

*_durations.txt

*_accdist_X_X_X_X.png

*_distance_X_X_X_X.png

*_duration_distplot_X_X_X_X.png

*_locomotion_X_X_X_X.png

*_speed_distplot_X_X_X_X.png

##### Validation part
*_pointed_X.mp4

\* can be specified with `--prefix` option, otherwise it is set as `stdout`. Four Xs included just before the extensions, inidicate attribute of the area analyzed. Additionaly, KaicoTracker can produce video files used for tracking, using `--liveSave` option.

---
### Options
###### I/O parameter
**--input file [file ...]**
Path to a video or a sequence of image. Multiple videos can be analyzed at the same time. Please provide them in order.

**--prefix prefix**
Prefix of path to output files.

**--fps FPS**
Output FPS, default is -1, which uses the same FPS with the input.

**--format {png,pdf}**
Output format for figures, default is 'png'.

**--videoFormat {'mp4', 'AVI'}**
Output format for videos, default is 'mp4'.

**--NoShrink**
If specified, videos are generated with original size. Otherwise, they are resized into 1/4.

###### Tracking parameter
**--algo {MOG2,KNN}**
Background subtraction method, default is KNN.

**--learningRate float**
Learning Rate for applied method, default is 0.8.

**--blurringSquare int px**
Edge length of blurring square in px.

###### Aanalyzing parameter
**--lapse seconds**
Time-lapse interval, default is 3 s.

**--segment int px**
Length of segment used for area segmentation in px.

**--segmentEdgeLength int px**
Length of segment square in px.

**--window frames**
Seed window for duration analysis, default is 30 s.

###### Parameter specifies what to be analyzed
**--analysisRange start end**
A range of frames to be analyzed in the input video.

**--complexBorder**
If specified, borders are determined by two-round calculation

**--autoCrop columns rows**
Set coloumns and rows for cropping.

**--autoCropPreAnalysis Preanalyezed_frames**
The number of frames analyzed for determining cropped area. `--autoCrop` must be specified.

**--cropArea bottom top left right**
Cropping area of video in px.

**--cropThreshold ratio**
Threshold to cut inappropreate borders, default=0.01. If the border separate the points whose ratio is below the designated value, the border is discarded.

###### Selection of stages
**--noPoint**
If specified, validation stage is skipped.

**--skipTracking**
If specified, tracking stage is skipped, `--trackingResult` must be specified to designate the result to be analyzed.

**--onlyTracking**
If specified, only tracking stage is run.

**--onlyAnalyis**
If specified, only Analysis stage is run.

**--onlyPoint** 
If specified, only validation stage is run.

###### Parameter required for specific stages
**--cropAreaForPoint bottom top left right**
Cropping position of frame in px, required when `--skipTracking` is specified and `--noPoint` is not specified, or `--onlyPoint` is specified

**--trackingResult file**
Path to output files of Tracking. If not specified, {prefix}.txt is read.

**--analysisResult file**
Path to output files of Analysis. If not specified, {prefix}_positions.txt is read.

###### Display parameter
**--live**
If specified, the videos under proccessing are displayed.

**--liveSave**
If specified, the videos under proccessing are saved. Mainly used for developmental purpose.

###### Miscellaneous
**-h, --help**
Show the help message and exit.

---
### References
under construction
