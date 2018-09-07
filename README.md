This project is a replication of the work from CAI 2016 - Batch Mode Active Learning for Regression With Expected Model Change.


**normalize_data.py** - This both normalizes the format and the data.  In normalizing the format, it pickles an object containing data, target, feature_names, target_names.  It also goes through all the features and normalizes following page 56 of Cai 2013 - Maximizing Expected Model.

**process_al.py** - Run active learning models.

**plot.py** - Test different plotting options.

**test.py** - Test suit to make sure code is working the as expected.

**timer.py** - Timer class used to measure speed of different parts of the code.  Used to optimize code segments.
