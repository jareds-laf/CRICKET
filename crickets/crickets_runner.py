import argparse
from blimpy import calcload, Waterfall
import time
import os
import numpy as np
from analysis import write_output_table, normalize_path, plot_exkurt, plot_tavg_power
import glob
import pandas as pd
import logging

##### Setup parser #####

parser = argparse.ArgumentParser(
                    description='Flag RFI heavy frequency channels based on the excess kurtosis of each channel.')

# Input/analysis arguments
parser.add_argument('--input_file',
		    help='(Required) Path to input filterbank file (including file name).', 
			type=str,
			required=True)
parser.add_argument('--output_file',
		    help='(Required) Path to output csv file (optionally including file name).',
			type=str,
			required=True)
parser.add_argument('--info_table_loc', '-i',
		    help='(Required) Directory to find info tables.',
			type=str,
			required=True)
parser.add_argument('--threshold', '-t',
		    help='(Required) Minimum value of excess kurtosis used to flag channels with significant RFI. Can be any decimal number.',
			type=float,
			default=5,
			required=True)
parser.add_argument('--ndivs', '-n',
		    help='(Required) Number of frequency bins to split waterfall object into. Can be any integer.',
			type=int,
		    default=256,
			required=True)
parser.add_argument('--all_freqs', '-a',
		    help='(Optional) Choose whether or not to include all frequency bins in the output table, even if they are not flagged as RFI. If this is not specified, only the flagged bins will be included in the output table.',
		    action='store_true',
			required=False)

# Plotting arguments
# TODO: Better implementation of the plotting arguments!
parser.add_argument('--plot', '-p',
		    help='(Optional) Choose whether or not to generate time-averaged power spectrum and excess kurtosis vs. frequency plots. Give output path for plots here (NOT including file name).',
			# action='store_true',
			type=str,
			required=False)
parser.add_argument('--plot_file_types', '--pft',
			help='(Optional, unless -p is given) Output file types (can be pdf, png, and/or jpg). Specify as many of these as you want!',
			choices=['png', 'pdf', 'jpg'],
			nargs='+',
			required=False)
# parser.add_argument('-p', '--plot_types',
# 		    help='(Optional) List of plot types. tavg_pwr: Time-averaged power spectrum. exkurt: Excess kurtosis vs. frequency plot.',
# 			choices=['exkurt', 'tavg_pwr'],
# 			nargs='+',
# 			# type=str,
# 			# default=[None, None],
# 			required=False)
# TODO: Figure out how to implement custom plot bounds
# parser.add_argument('--plot_bnds',
# 		    help='(Optional) x and y bounds for plots.',
# 			required=False)

# Miscellaneous arguments
parser.add_argument('--verbose', '-v',
		    help='(Optional) Print more information about the input variables and the processes currently running.',
			action='store_true',
			required=False)

args = parser.parse_args()

##### Setup logging #####

# Create logger and console handler
logger = logging.Logger('crickets_runner')
logger.propagate = False
ch = logging.StreamHandler()

# Create formatter
formatter = logging.Formatter('%(asctime)s :: %(levelname)s :: %(name)s, %(funcName)s :: line %(lineno)d :: %(message)s')

# Add formatter to ch
ch.setFormatter(formatter)

# Add ch to logger
logger.addHandler(ch)

# Set logging level based on whether or not --verbose (-v) is specified
if args.verbose:
	logger.setLevel(logging.INFO)
	ch.setLevel(logging.INFO)
else:
	logger.setLevel(logging.WARNING)
	ch.setLevel(logging.WARNING)

##### Main code #####

# Normalize specified paths
filfil = normalize_path(args.input_file)
out_path = normalize_path(args.output_file)
itloc = normalize_path(args.info_table_loc)

# Check for valid filterbank file and output location
if (filfil[len(filfil) - 4:] != '.fil'):
	parser.error(f'Input file must be a valid filterbank (.fil) file. Specified input file path: {filfil}')
if (os.path.isfile(filfil) == False):
	parser.error(f'Input file cannot be found. Make sure to include the file name in the path. Specified input file: {filfil}')

# Check to see if info table location is valid
# if (os.path.isdir(itloc) == False):
# 	parser.error(f'Info table location is not a valid directory. Specified info table location: {itloc}')

# TODO: Figure out how to check for valid output filepath (might not need to do this)
# Note: Data type checking is automatically covered, so long as you specify a type= ! :D

if os.path.isdir(out_path):
	if out_path[-1] != '/':
		out_path += '/'
		logger.debug("The output file path is a directory.")
		output_file_loc = out_path + f"crickets_{args.input_file[args.input_file.rfind('/')+1:len(args.input_file)-4]}_{args.ndivs}_{args.threshold}.csv"
	else:
		logger.debug("The output file path is a directory.")
		output_file_loc = out_path + f"crickets_{args.input_file[args.input_file.rfind('/')+1:len(args.input_file)-4]}_{args.ndivs}_{args.threshold}.csv"
if (os.path.isfile(out_path)) | (out_path[-4:] == '.csv'):
	# print("The output file is a file.")
	output_file_loc = out_path
	


# info_table_list = glob.glob(os.path.join(itloc, f'info_table*.csv'))
# print(f"\n\nHere is the info_table_list: {info_table_list}\n\n")
# if info_table_list == []:
# 	parser.error(f"No info tables found in {itloc}. Make sure you ran info_table_gen.py, and that the info tables are in the correct directory and are named correctly.")
# logger.debug(f"\ninfo_table_list: {info_table_list}")
# info_table = pd.read_csv(itloc)

# Run analysis code and generate the output tables for each info table
# for it in info_table_list:
it = args.info_table_loc
logger.info(f"\nit from runner: {it}, {type(it)}\n")
info_table = pd.read_csv(it)
freqs = info_table['freq']
pows = info_table['tavg_power']


write_output_table(info_table=it, output_filepath=output_file_loc, n_divs=args.ndivs, threshold=args.threshold, all=args.all_freqs)
logger.info(f'Output table generated at {output_file_loc}')
# TODO: Check to see if plot output and plot file types are given if -p is specified

# Plotting code
if args.plot:
		
		logger.debug(f"Freqs:\n {freqs}, {type(freqs)}")

		f_min = np.floor(np.ceil(freqs.min()))
		f_max = np.ceil(np.ceil(freqs.max()))
		p_min = np.floor(np.floor(pows.min()))
		p_max = np.ceil(np.floor(pows.max()))

		logger.debug(f"Maximums: {f_max}, {p_max}")
		logger.debug(f"Minmums: {f_min}, {p_min}")

		logger.debug(f"\nFrequency range: {f_max - f_min} MHz")
		logger.debug(f"\nPower range: {p_max - p_min} counts\n")

		if args.input_file.rfind('/') != -1:
			name_index_start = args.input_file.rfind('/') + 1
		else:
			name_index_start = 1
		name_index_end = args.input_file.rfind('.')
		
		# Excess kurtosis plot
		logger.info("Generating exkurt plot...")
		exkurt_plot_name = f'plot_exkurt_{args.input_file[name_index_start:name_index_end]}_{args.ndivs}_{args.threshold}'
		# plot_names = []
		# for i in args.plot_file_types:
		# 	exec(f"exkurt_plot_name_{i} = f'plot_exkurt_{args.input_file[name_index_start:name_index_end]}_{args.ndivs}_{args.threshold}.{i}'")
		# 	exec(f"plot_names.append(exkurt_plot_name_{i})")
		# 	print()
		# 	print("plot names item")
		# 	exec(f"logger.info(plot_names.append(exkurt_plot_name_{i}))")
		# 	print()
		# 	print()

		plot_exkurt(info_table=it, n_divs=args.ndivs, threshold=args.threshold,
				unfiltered=True, clean_chnls=True, rfi=True,
				f_start=f_min, f_stop=f_max,
				output_dest=normalize_path(args.plot),
				output_type=args.plot_file_types)
			# TODO: Once you figure out how to do plot boundaries, put in k_start and k_stop! :D
		
		# for i in plot_names:
		# 	exec(f"logger.info(f'exkurt plot generated at {os.path.join(args.plot, i)}')")

		# Time-averaged power spectrum plot
		print()
		logger.info("Generating tavg_power plot...")
		tavg_pwr_plot_name = f'plot_tavg_pwr_{args.input_file[name_index_start:name_index_end]}_{args.ndivs}_{args.threshold}'
		# plot_names = []
		# for i in args.plot_file_types:
		# 	exec(f"tavg_pwr_plot_name_{i} = f'plot_tavg_pwr_{args.input_file[name_index_start:name_index_end]}_{args.ndivs}_{args.threshold}.{i}'")
		# 	exec(f"plot_names.append(tavg_pwr_plot_name_{i})")

		plot_tavg_power(info_table=it, n_divs=args.ndivs, threshold=args.threshold,
				f_start=f_min, f_stop=f_max,
				p_start=p_min, p_stop=p_max, show_filtered_bins=True,
				output_dest=normalize_path(args.plot),
				output_type=args.plot_file_types)
		
		# for i in plot_names:
		# 	gen_at_loc = normalize_path(f"{args.plot}.{i}")
		# 	exec(f"logger.info(f'tavg_pwr plot generated at {gen_at_loc}')")

# TODO: Fix the way verbose is handled to use logging/logger!
# Verbose outputs: Print argument values in case something isn't working :)
print()
logger.info("Argument values:")
logger.info(f"Input filename: {args.input_file}, {type(args.input_file)}")
logger.info(f"Output filename: {args.output_file}, {type(args.output_file)}")
logger.info(f"Threshold: {args.threshold}, {type(args.threshold)}")
logger.info(f"Number of bins: {args.ndivs}, {type(args.ndivs)}")
logger.info(f"All: {args.all_freqs}, {type(args.all_freqs)}")
logger.info(f"Plot (bool): {args.plot}, {type(args.plot)}")
logger.info(f"Plot file type(s): {args.plot_file_types}, {type(args.plot_file_types)}")
logger.info(F"Verbose: {args.verbose}, {type(args.verbose)}")