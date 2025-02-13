#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 10:00:04 2019

@author: Vince
"""

#This tutorial is divided into 10 parts; they are:
#
#1. Human Activity Recognition
#2. Activity Recognition Using Smartphones Dataset
#3. Download the Dataset
#4. Load Data
#5. Balance of Activity Classes
#6. Plot Time Series Data for One Subject
#7. Plot Histograms Per Subject
#8. Plot Histograms Per Activity
#9. Plot Activity Duration Boxplots
#10. Approach to Modeling

#1. Human Activity Recognition

# HAR for short, is the problem of predicting what a person is doing based on a trace of their movement using sensors. Movements are often normal indoor activities such as standing, sitting, jumping, and going up stairs. Sensors are often located on the subject, such as a smartphone or vest, and often record accelerometer data in three dimensions (x, y, z).
# Human Activity Recognition (HAR) aims to identify the actions carried out by a person given a set of observations of him/herself and the surrounding environment. Recognition can be accomplished by exploiting the information retrieved from various sources such as environmental or body-worn sensors.
# It is a challenging problem because there is no clear analytical way to relate the sensor data to specific actions in a general way. It is technically challenging because of the large volume of sensor data collected (e.g. tens or hundreds of observations per second) and the classical use of hand crafted features and heuristics from this data in developing predictive models.
# Sensor-based activity recognition seeks the profound high-level knowledge about human activities from multitudes of low-level sensor readings. Conventional pattern recognition approaches have made tremendous progress in the past years. However, those methods often heavily rely on heuristic hand-crafted feature extraction, which could hinder their generalization performance. […] Recently, the recent advancement of deep learning makes it possible to perform automatic high-level feature extraction thus achieves promising performance in many areas.

# HAR基於追蹤受測者行動的感測器，預測其正在進行的動作，包括正常的屋內行動，例如：站起來、坐下去、跳躍、上樓梯等。
# 感測器穿戴在受測者身上，記錄加速計(accelerometer)或陀螺儀(gyroscope)的三維數據。HAR的目的是透過一組受測者相對於環境的觀測值，辨識出受測的上述動作。
# HAR具挑戰性的原因是並無明確的解析關係，說明感測器數據與各種動作的關聯。此外，從技術上而言，每秒數以時計或百計的大量感測器資料，造成分析上的困難。過往手工調配的分析屬性，以及專家們的經驗法則，使得預測建模工作晦澀難懂。
# HAR從大量低階的感測器讀數，淬煉出人類行動的高階知識。除了傳統機器學習或型態辨識下手工調配屬性的方式，近年來的深度學習的進展，有可能自動完成高階的屬性萃取工作，因而在此領域取得有前景的發展。

#2. Activity Recognition Using Smartphones Dataset
# A standard human activity recognition dataset is the ‘Activity Recognition Using Smartphones‘ dataset made available in 2012.
#It was prepared and made available by Davide Anguita, et al. from the University of Genova, Italy and is described in full in their 2013 paper “A Public Domain Dataset for Human Activity Recognition Using Smartphones.” The dataset was modeled with machine learning algorithms in their 2012 paper titled “Human Activity Recognition on Smartphones using a Multiclass Hardware-Friendly Support Vector Machine.”
# 2012年義大利熱納亞大學的Davide Anguita等人以支援向量機建立HAR的多個分類模型，2013年他們將資料集公開釋出。

# The dataset was made available and can be downloaded for free from the UCI Machine Learning Repository: Human Activity Recognition Using Smartphones Data Set
# The data was collected from 30 subjects aged between 19 and 48 years old performing one of 6 standard activities while wearing a waist-mounted smartphone that recorded the movement data. Video was recorded of each subject performing the activities and the movement data was labeled manually from these videos.
# 現在我們可從UCI機器學習數據庫中下載人類行動辨識之智慧型手機資料集，該資料集由年紀19到48的30位受測者，在腰間穿戴智慧型手機時作出六個標準動作。

#The six activities performed were as follows:
#
#Walking
#Walking Upstairs
#Walking Downstairs
#Sitting
#Standing
#Laying
#The movement data recorded was the x, y, and z accelerometer data (linear acceleration) and gyroscopic data (angular velocity) from the smart phone, specifically a Samsung Galaxy S II.
#
#Observations were recorded at 50 Hz (i.e. 50 data points per second). Each subject performed the sequence of activities twice, once with the device on their left-hand-side and once with the device on their right-hand side.
#
#A group of 30 volunteers with ages ranging from 19 to 48 years were selected for this task. Each person was instructed to follow a protocol of activities while wearing a waist-mounted Samsung Galaxy S II smartphone. The six selected ADL were standing, sitting, laying down, walking, walking downstairs and upstairs. Each subject performed the protocol twice: on the first trial the smartphone was fixed on the left side of the belt and on the second it was placed by the user himself as preferred
#
#— A Public Domain Dataset for Human Activity Recognition Using Smartphones, 2013.
#
#The raw data is not available. Instead, a pre-processed version of the dataset was made available.
#
#The pre-processing steps included:
#
#Pre-processing accelerometer and gyroscope using noise filters.
#Splitting data into fixed windows of 2.56 seconds (128 data points) with 50% overlap.
#Splitting of accelerometer data into gravitational (total) and body motion components.
#These signals were preprocessed for noise reduction with a median filter and a 3rd order low-pass Butterworth filter with a 20 Hz cutoff frequency. […] The acceleration signal, which has gravitational and body motion components, was separated using another Butterworth low-pass filter into body acceleration and gravity.
#
#— A Public Domain Dataset for Human Activity Recognition Using Smartphones, 2013.
#
#Feature engineering was applied to the window data, and a copy of the data with these engineered features was made available.
#
#A number of time and frequency features commonly used in the field of human activity recognition were extracted from each window. The result was a 561 element vector of features.
#
#The dataset was split into train (70%) and test (30%) sets based on data for subjects, e.g. 21 subjects for train and nine for test.
#
#This suggests a framing of the problem where a sequence of movement activity is used as input to predict the portion (2.56 seconds) of the current activity being performed, where a model trained on known subjects is used to predict the activity from movement on new subjects.
#
#Early experiment results with a support vector machine intended for use on a smartphone (e.g. fixed-point arithmetic) resulted in a predictive accuracy of 89% on the test dataset, achieving similar results as an unmodified SVM implementation.
#
#This method adapts the standard Support Vector Machine (SVM) and exploits fixed-point arithmetic for computational cost reduction. A comparison with the traditional SVM shows a significant improvement in terms of computational costs while maintaining similar accuracy […]
#
#— Human Activity Recognition on Smartphones using a Multiclass Hardware-Friendly Support Vector Machine, 2012.
#
#Now that we are familiar with the prediction problem, we will look at loading and exploring this dataset.






#3. Download the Dataset
#The contents of the “train” and “test” folders are similar (e.g. folders and file names), although with differences in the specific data they contain.

#train和test資料夾內容相似

#Inspecting the “train” folder shows a few important elements:
#
#An “Inertial Signals” folder that contains the preprocessed data.
#The “X_train.txt” file that contains the engineered features intended for fitting a model.
#The “y_train.txt” that contains the class labels for each observation (1-6).
#The “subject_train.txt” file that contains a mapping of each line in the data files with their subject identifier (1-30).

#train資料夾中包括：預處理好的慣性訊號資料夾Inertial Signals、為了配適模型以調整好的屬性X_train.txt、訓練集類別標籤y_train.txt、受測者編號與觀測值的對應subject_train.txt

#The “Inertial Signals” directory contains 9 files.
#
#Gravitational acceleration data files for x, y and z axes: total_acc_x_train.txt, total_acc_y_train.txt, total_acc_z_train.txt.
#Body acceleration data files for x, y and z axes: body_acc_x_train.txt, body_acc_y_train.txt, body_acc_z_train.txt.
#Body gyroscope data files for x, y and z axes: body_gyro_x_train.txt, body_gyro_y_train.txt, body_gyro_z_train.txt.
#The structure is mirrored in the “test” directory.

# 預處理好的慣性訊號資料夾Inertial Signals裡包括：
# x, y, z三軸的重力加速度資料：total_acc_x_train.txt, total_acc_y_train.txt, total_acc_z_train.txt
# x, y, z三軸的身體加速資料：body_acc_x_train.txt, body_acc_y_train.txt, body_acc_z_train.txt
# x, y, z三軸的身體陀螺儀資料：body_gyro_x_train.txt, body_gyro_y_train.txt, body_gyro_z_train.txt
# test目錄中的結構類似

#We will focus our attention on the data in the “Inertial Signals” as this is most interesting in developing machine learning models that can learn a suitable representation, instead of using the domain-specific feature engineering.
#
#Inspecting a datafile shows that columns are separated by whitespace and values appear to be scaled to the range -1, 1. This scaling can be confirmed by a note in the README.txt file provided with the dataset.
#
#Now that we know what data we have, we can figure out how to load it into memory.

# 後續焦點放在Inertial Signals資料夾，因為可由此運用機器學習模型，學習合宜的知識表達而非利用特定領域人工調配的屬性
# 各資料檔中各個欄位以空白符分隔，且數值量綱已調整到-1與1之間

#4. Load Data
# load dataset
from numpy import dstack
from pandas import read_csv

# load a single file as a numpy array
def load_file(filepath):
	dataframe = read_csv(filepath, header=None, delim_whitespace=True)
	return dataframe.values

data = load_file('./data/UCI HAR Dataset/train/Inertial Signals/total_acc_y_train.txt')
print(data.shape)

# load a list of files, such as x, y, z data for a given variable
def load_group(filenames, prefix=''):
	loaded = list()
	for name in filenames:
		data = load_file(prefix + name)
		loaded.append(data)
	# stack group so that features are the 3rd dimension
	loaded = dstack(loaded)
	return loaded

# load the total acc data
filenames = ['total_acc_x_train.txt', 'total_acc_y_train.txt', 'total_acc_z_train.txt']
total_acc = load_group(filenames, prefix='./data/UCI HAR Dataset/train/Inertial Signals/')
print(total_acc.shape)

#5. Balance of Activity Classes


#6. Plot Time Series Data for One Subject


#7. Plot Histograms Per Subject


#8. Plot Histograms Per Activity


#9. Plot Activity Duration Boxplots


#10. Approach to Modeling

### Refernce:
# How to Model Human Activity From Smartphone Data
# https://machinelearningmastery.com/how-to-model-human-activity-from-smartphone-data/
