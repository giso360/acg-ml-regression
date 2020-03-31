# acg-ml-regression

# Data Source
-----------
PT and RB data found in: https://fairmodel.econ.yale.edu/vote2012/affairs.txt
[fair book pg. 71]
Data gathered from two magazine surveys that were both conducted via mail:
1) Psychology Today (1969) 
2) Redbook (1974/ American women's magazine that publishes since 1903 https://en.wikipedia.org/wiki/Redbook) - Women only data ==> SOS: frequency of gender="female"

Dataset exclusion criteria:
- People who were married more than once were excluded
- Unemployed people were also excuded
- Excluded people that failed to answer all questions



Run Instructions
----------------

-  In ..seaborn\algorithms.py change L84 to: resampler = integers(0, n, n, dtype=np.int_)
