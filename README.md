# CRICKETS

#### **Please note that this readme is not up-to-date. The instructions on how to use CRICKETS from the command line, in particular, are outdated. However, one should be able to figure out the steps to use it with the command**

```
python3 [path/to/rfi_id.py] -h
```

CRICKETS (Categorization of RFI In COSMIC with Kurtosis for Extraterrestrial Searches) is a packaged designed to flag heavy RFI frequency bins in data that comes from [COSMIC](https://science.nrao.edu/facilities/vla/observing/cosmic-seti). This is accomplished by generated a time-averaged power spectrum from an input filterbank or hdf5 file and analyzing the excess kurtosis ($exkurt$) of the time-averaged power in a specified number of frequency bins.

In its current state, this package is **NOT** for differentiating any signals of scientific interest from RFI. Its strength is combing through observations of sources with little to no fine frequency emissions that could be mistaken for RFI. Assuming noise follows a Gaussian distribution, any frequency bins with an excess kurtosis outside of a specifiable range around 0 are likely RFI. With the limitations of this program in mind, the best use of this package is to flag frequency ranges that are heavy in RFI and masking these frequencies after the data from the primary observations have been collected. The package is also effective for studying the overall RFI environment of a series of observations at different times.

*A note on terminology: With the typical (Pearson) definition of kurtosis, a Gaussian distribution has a kurtosis of 3 and an "excess kurtosis" of 0. This package uses the Pearson definition of kurtosis as of July 18th, 2023.*

# Installation:
As of May 20th, 2024, the package is still a work in progress and it is not entirely functional. Once completed, there will be two primary ways to install the package.

## Dependencies
The versions of the following required packages shown are simply a snapshot of the versions used in the development of this package. If there are any issues with this package, try installing these specific versions of the code as a troubleshooting step. Otherwise, install any version and the code *shoud* work.

- blimpy==2.1.4
- matplotlib==3.8.1
- numpy==1.26.2
- pandas==2.1.2
- scipy==1.11.3

## Install by cloning the repository
Find or create the folder you would like to clone this repository to, then use the following command to clone via HTTPS:

```
git clone https://github.com/jareds-laf/CRICKETS.git
```

Alternatively, you can use this command to clone via SSH:

```
git clone git@github.com:jareds-laf/CRICKETS.git
```

## Install via pip (WIP)
This package is currently available on [Test PyPI](https://test.pypi.org/project/crickets/). It can be installed with the following command:

```
pip install -i https://test.pypi.org/simple/ crickets
```

# Summary of the Process
As of July 28th, 2023, the functionality of this package *most* works! We are currently in the process of testing everything to ensure regular operations. The following is a summary of the process of running this package that is expected to be accurate once the package is fully functional.

## Flow of Analysis
An input filterbank file is used to generate a [blimpy](https://github.com/UCBerkeleySETI/blimpy) waterfall object. This can be done for an entire folder of filterbank files. The power is then averaged over the time domain, and this time-averaged power and the frequencies in the filterbank are output to a .csv file. This .csv file is then read by the second part of the package, which runs the primary analysis. The time-averaged power is split into a specifiable number of frequency bins. The default number of bins is 256. So as to avoid any infinite excess kurtosis, the data is rescaled with a simple division of each time-averaged power value by $1*10^{9}$. Then, the excess kurtosis of each bin is then calculated using [scipy](https://github.com/scipy/scipy), and bins with a high* excess kurtosis are flagged. The data is once again checked for any unwanted infinities. The high RFI bins are output in a .csv file with the following columns:

- **rfi_bin_bots**: High RFI frequency bin bottoms
- **rfi_bin_tops**: High RFI frequency bin tops
- **exkurt**: Excess kurtosis of corresponding bin

**The minimum threshold to flag high excess kurtosis bins can be specified by the user. The threshold constitutes the ends of the range of kurtoses that are considered low. In other words, all flagged bins satisfy this condition:*

$|exkurt| \geq threshold$

The user may also choose to include all frequency bins in the output file so that they can determine the minimum excess kurtosis threshold without studying either of the plots that can be generated.

One might use a higher threshold if they are using this package as a quick and dirty way to flag problematic frequency ranges. A lower threshold can be determined with some playing around, though this is not always necessary.

### Plotting Functions
The user can choose to generate two plots to aid them in their scientific endeavors using the ```--plot``` or ```-p``` option, or by importing the necessary functions into a Python script or notebook (WIP). The two types of plots are:
1. exkurt: Plot the excess kurtosis of each bin against their corresponding bin bottoms. One can choose to include up to 3 of the following data categories to plot the excess kurtosis of:
   1. Unfiltered data, denoted by black circles
   2. "Clean" (low RFI) channels, denoted by smaller green circles
   3. "Dirty" (high RFI) channels, denoted by smaller red circles
2. tavg_power: Plot the time-averaged power spectrum (i.e., time-averaged power vs. frequency). The user can specify whether or not to show the bins that have been flagged as having heavy RFI. Flagged bins are shown with transparent red rectangles that span the height of the graph and are "behind" the main spectrum so as not to steal the show. In particularly RFI-dominant regions, it may be difficult to see that the problematic bins were flagged. In this case, the user is encouraged to change the bounds of the y-axis to get a better view of the flagged bins, or to look at the first type of plot to see the flagged bins more clearly.
    
# Usage
## From the Command Line
All functions of this package can be performed from the command line. Note that the processing is done in two stages using two different .py files. Thus, there are at least two commands that must be run.

The first command runs ```info_table_gen.py```, which generates the table containing the frequencies and time-averaged power of the input filterbank file. The most general syntax is as follows:

```
python3 <path/to/info_table_gen.py> [Options]
```

#### Options:
- ```wf_path```, ```-w``` (Required) Path to directory containing all input filterbank files.
- ```--info_table_loc```, ```-i``` (Required) Directory to save info tables to.

The second command runs ```crickets_runner.py```, which runs the primary analysis.

The most general syntax is as follows:

```
python3 <path-to-crickets_runner.py> [Options]
```



#### Options:
- ```--input_file``` (Required) Path to input filterbank file (including file name).
- ```--output_file``` (Required) Path to output csv file containing frequency bin bottoms, frequency bin tops, and excess kurtosis of each bin (optionally including file name).
- ```--info_table_loc```, ```-i``` (Required) Directory to find info tables.
- ```--threshold```, ```-t``` (Required) Minimum value of excess kurtosis used to flag channels with significant RFI. Can be any decimal number.
- ```--ndivs```, ```-n``` (Required) Number of frequency bins to split waterfall object into. Can be any integer.
- ```--all_freqs```, ```-a``` (Optional) Choose whether or not to include all frequency bins in the output table, even if they are not flagged as RFI. If this is not specified, only the flagged bins will be included in the output table.
- ```--plot```, ```-p``` (Optional) Choose whether or not to generate time-averaged power spectrum and excess kurtosis vs. frequency plots. Give output path for plots here (NOT including file name).
- ```plot_file_types```, ```--pft``` (Optional, unless -p is given) Output file types (can be pdf, png, and/or jpg). Specify as many of these as you want! 
- ```--verbose```, ```-v``` (Optional) Print more information about the input variables and the processes currently running.

For both ```info_table_gen.py``` and ```crickets_runner.py```, the user can specify the ```--help``` or  ```-h``` options to list all options, as well!

## As a Python Package
WIP!

# Examples
## Command Line Usage
### Running analysis from the command line without plots

The template for running analysis without generating plots is as follows:

```
python3 <path/to/crickets_runner.py> --input_file <path/to/filterbank.fil> --info_table_loc <path/to/info_table_filterbank.csv> --output_filename <path/to/output.csv> -t <kurtosis_threshold> -n <number_of_bins>
```

Here is an example with a minimum excess kurtosis threshold of 5 where the waterfall object gets broken into 256 frequency bins.

```
python3 /home/alice/CRICKETS/crickets/crickets_runner.py --info_table_loc /home/alice/info_tables/info_table_filterbank1.csv --input_file /home/alice/filterbank/filterbank1.fil --output_file /home/alice/crickets_output/tables -t 5 -n 256
```

Here is a screenshot of the output table taken in Microsoft Excel:

<img src="https://github.com/jareds-laf/CRICKETS/blob/main/examples/example_output_excel.png" alt="Example table output. This table can be found at example/example_output_excel.png" width="227" height="350" />

### Running analysis from the command line with plots
***WIP!***

The template for running analysis and generating plots is as follows:

```
python3 <path/to/crickets_runner.py> --input_file <path/to/filterbank.fil> --info_table_loc <path/to/info_table_filterbank.csv> --output_file <path/to/output.csv> -t <exkurt_threshold> -n <number_of_bins> -p <path/to/plotting_directory> --plot_file_types <tuple containg 'png', 'jpg', and/or 'pdf'>
```

Here is the same example as above (minimum excess kurtosis threshold of 5 with 256 frequency bins), this time generating the two plots.

```
python3 /home/alice/CRICKETS/crickets/crickets_runner.py --info_table_loc /home/alice/info_tables/info_table_filterbank1.csv --input_file /home/alice/filterbank/filterbank1.fil --output_file /home/alice/crickets_output/tables -t 5 -n 256 -p /home/alice/plots --plot_file_types png pdf jpg
```

Here are some screenshots of the resulting plots.

.png:
WIP!
.jpg:
WIP!
.pdf:
WIP!
## Usage as a Package
WIP!
```
from crickets.analysis import get_exkurt, write_output_table

from crickets.plotting import plot_tavg_power, plot_exkurt
```