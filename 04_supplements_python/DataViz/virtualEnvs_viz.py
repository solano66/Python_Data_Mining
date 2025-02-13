### Please try the followings on your Anaconda Prompt (Don't forget use the system administrator to turn on Anaconda Prompt)

### Check how many virtual envs you have first
'''
(base) Ching-Shihs-MBP:~ Vince$ conda info --envs

or

(base) Ching-Shihs-MBP:~ Vince$ conda info -e
'''

### Create the virtual env 'viz' and specify the Python version
'''
(base) Ching-Shihs-MBP:~ Vince$ conda create -n viz python=3.6 anaconda --y
'''

### Activate the virtual env 'viz'
'''
(base) Ching-Shihs-MBP:~ Vince$ conda activate viz

or

(base) Ching-Shihs-MBP:~ Vince$ source activate viz (for older conda versions)
'''

### Install specific version of packages you need in the virtual env 'viz' (remove the newest packages automatically)
'''
(viz) Ching-Shihs-MBP:~ Vince$ conda install matplotlib==3.1.3 pandas==0.22.0 --y
'''

### Install packages 'ggplot' and 'plotnine' from conda-forge
'''
(viz) Ching-Shihs-MBP:~ Vince$ conda install -c conda-forge ggplot plotnine --y
'''

### Start the IDE Spyder (conda install spyder==4 --y if it is not available)
'''
(viz) Ching-Shihs-MBP:~ Vince$ spyder

or

(viz) Ching-Shihs-MBP:~ Vince$ spyder --new-instance
'''

### After starting your Spyder in the virtual env, you can check where you are now ....

import sys
sys.prefix

### Check if proper versions of related packages installed

!conda list ggplot # or plotnine, pandas, matplotlib

### Deactivate virtual env 'viz'
'''
(viz) Ching-Shihs-MBP:~ Vince$ conda deactivate

or

(viz) Ching-Shihs-MBP:~ Vince$ source deactivate (for older conda versions)
'''

### Maybe remove virtual env 'viz' someday
'''
(base) Ching-Shihs-MBP:~ Vince$ conda remove -n viz --all
'''

### References:
# Create virtual environments for python with conda
# https://uoa-eresearch.github.io/eresearch-cookbook/recipe/2014/11/20/conda/


