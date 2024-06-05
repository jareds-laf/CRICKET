# CRICKETS

#### **Please note that this readme is not up-to-date. The instructions on how to use CRICKETS from the command line, in particular, are outdated. However, one should be able to figure out the steps to use CRICKET with the command**

```
python3 [path/to/rfi_id.py] -h
```

CRICKETS (Categorization of RFI In COSMIC with Kurtosis for Extraterrestrial Searches) is a packaged designed to flag heavy RFI frequency bins in data that comes from [COSMIC](https://science.nrao.edu/facilities/vla/observing/cosmic-seti). This is accomplished by generated a time-averaged power spectrum from an input filterbank or hdf5 file and analyzing the excess kurtosis ($exkurt$) of the time-averaged power in a specified number of frequency bins.

This package is **NOT** for differentiating any signals of scientific interest from RFI. Its strength is combing through observations of sources with little to no fine frequency emissions that could be mistaken for RFI. Assuming noise follows a Gaussian distribution, any frequency bins with an excess kurtosis outside of a specifiable range centered around 0 are likely RFI. With the limitations of this program in mind, the best use of this package is to flag frequency ranges that are heavy in RFI and passing this information to other parts of the post-processing pipeline so that resources are not spent analyzing RFI-dominated frequencies.

*A note on terminology: This package uses the Pearson definition of kurtosis, meaning a Gaussian distribution has a kurtosis of 3 and an excess kurtosis of 0.*

# Installation:
As of June 5th, 2024, the package is still a work in progress and it is not entirely functional. It can still be installed using the following steps.

## Dependencies
The versions of the following required packages shown are a snapshot of the versions used in development. If you encounter any issues, consider installing these specific versions of the dependencies as a troubleshooting step:

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

# Summary of the Process
As of June 5th, 2024, CRICKETS is still in development. The following is a summary of how it will run once development is complete. Any functions that have yet to be implemented are noted.

## Flow of Analysis
An input filterbank or hbf5 file is used to generate a [blimpy](https://github.com/UCBerkeleySETI/blimpy) waterfall object. The power is then averaged with respect to time, and this time-averaged power and their respective frequencies are tabulated. The time-averaged power is split into a specifiable number of frequency bins (default = 256). Each time-averaged power value is divided by $1*10^{9}$ so as to avoid any infinite excess kurtosis values. 

Then, the excess kurtosis of each bin is then calculated using [scipy](https://github.com/scipy/scipy), and bins with a high[^1] excess kurtosis are flagged.High RFI bins are output to a .csv file with the following columns:

- **exkurt**: Excess kurtosis of corresponding bin
- **rfi_bin_bots**: High RFI frequency bin bottoms
- **rfi_bin_tops**: High RFI frequency bin tops

[^1]: *The minimum threshold to flag high excess kurtosis bins can be specified by the user (default threshold = 5). Flagged bins satisfy this condition:$*

$|exkurt| \geq threshold$

[DEPRECATED, TO RETURN] The user may also choose to include all frequency bins in the output file so that they can determine the minimum excess kurtosis threshold without studying either of the plots that can be generated.

One might use a higher threshold if they are using this package as a quick and dirty way to flag problematic frequency ranges. A lower threshold can be determined with some playing around, though this is not always necessary.

### Plotting Functions
The user can choose to generate two plots using the ```--plot``` or ```-p``` option. The two types of plots are:
1. exkurt: Plot the excess kurtosis of each bin against their corresponding bin bottoms. One may include up to 3 of the following data categories to plot the excess kurtosis of:
   1. Unfiltered data, denoted by black circles
   2. "Clean" (low RFI) channels, denoted by smaller green circles
   3. "Dirty" (high RFI) channels, denoted by smaller red circles
2. tavg_power: Plot the time-averaged power spectrum (i.e., time-averaged power vs. frequency). The user can specify whether or not to show the bins that have been flagged as having heavy RFI. Flagged bins are shown with transparent red rectangles that span the height of the graph.
    
# Usage
All functions of this package can be performed from the command line.

The first command runs ```info_table_gen.py```, which generates the table containing the frequencies and time-averaged power of the input filterbank file. The most general syntax is as follows:

```
python3 <path/to/rfi_id.py> [Options]
```

#### Options:
- ```--input_file```, ```-w``` (Required) Path to input filterbank or hdf5 file (including file name).
- ```--output_file``` (Required) Path to output csv file containing excess kurtosis, frequency bin bottoms, and frequency bin tops (optionally including file name).
- ```--threshold```, ```-t``` (Required) Minimum value of excess kurtosis used to flag channels with significant RFI. Can be any decimal number.
- ```--ndivs```, ```-n``` (Required) Number of frequency bins to split waterfall object into. Can be any integer.
- ```--plot```, ```-p``` (Optional) Choose whether or not to generate time-averaged power spectrum and excess kurtosis vs. frequency plots. Give output path for plots here (NOT including file name).
- ```plot_file_types```, ```--pft``` (Optional, unless ```-p``` is given) Output file types (can be pdf, png, and/or jpg). Specify as many of these as you want! 
- ```--verbose```, ```-v``` (Optional) Print more information about the processes currently running.

One can always specify the ```--help``` or  ```-h``` options to list all options, as well!

# Examples
## Command Line Usage
### Running analysis from the command line without plots

The template for running analysis without generating plots is as follows:

```
python3 <path/to/rfi_id.py> --input_file <path/to/filterbank.fil> --output_file <path/to/output.csv> -t <kurtosis_threshold> -n <number_of_bins>
```

Here is an example with an excess kurtosis threshold of 3 where the waterfall object gets broken into 128 frequency bins.

```
python3 /home/alice/CRICKETS/crickets/rfi_id.py --input_file /home/alice/filterbank/mishmish_telescope_sband.fil --output_file /home/alice/crickets_output/tables -t 3 -n 128
```

[OUTDATED, TO BE REPLACED] Here is a screenshot of an example output table taken in Microsoft Excel:

<img src="https://github.com/jareds-laf/CRICKETS/blob/main/examples/example_output_excel.png" alt="Example table output. This table can be found at example/example_output_excel.png" width="227" height="350" />

### Running analysis from the command line with plots
***WIP!***

The template for running analysis and generating plots is as follows:

```
python3 <path/to/rfi_id.py> --input_file <path/to/filterbank.fil> --output_file <path/to/output.csv> -t <kurtosis_threshold> -n <number_of_bins> -p <path/to/plotting_directory> --plot_file_types <tuple containg 'png', 'jpg', and/or 'pdf'>
```

Here is the same example as above (minimum excess kurtosis threshold of 3 with 128 frequency bins), this time generating the two plots.

```
python3 /home/alice/CRICKETS/crickets/rfi_id.py --input_file /home/alice/filterbank/mishmish_telescope_sband.fil --output_file /home/alice/crickets_output/tables -t 3 -n 128 -p /home/alice/filfil/plots --plot_file_types png pdf
```

[WIP] Here are some screenshots of the resulting plots.

.png:

.pdf:
