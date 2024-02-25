# opt_proj2
This repository contains data analysis code for the paper "Ecosystem water-saving time scale varies spatially with typical drydown length" by Holtzman, Sloan, Potkay, Katul, Feng, and Konings (submitted 2023).

To replicate the results in the paper, first download half-hourly/hourly and daily data from FLUXNET2015 (https://fluxnet.org/data/fluxnet2015-dataset/), and LAI data from  Ukkola et al. (https://essd.copernicus.org/articles/14/449/2022/). 

Then run the scripts in the following order. You will need to make sure the file paths in the scripts point to the dataset locations on your computer.
1. process_daily_fluxnet.py
2. process_hourly_fluxnet.py
3. estimate_tau.py
4. gpp_curve_example.py
5. compare_medlyn_theory3.py
