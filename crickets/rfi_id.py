import argparse
from blimpy import calcload, Waterfall
import time
import os
import numpy as np
import glob
import pandas as pd
import logging
import matplotlib.pyplot as plt
from scipy.stats import norm, kurtosis
import numpy.ma as ma

# Setup parser

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

# Plotting arguments
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

# Miscellaneous arguments
parser.add_argument('--verbose', '-v',
		    help='(Optional) Print more information about the input variables and the processes currently running.',
			action='store_true',
			required=False)

args = parser.parse_args()

# Set up logger
logger = logging.getLogger('analysis')
if args.verbose:
    logger.setLevel(logging.DEBUG)
else:
    logger.setLevel(logging.INFO)

logger.propagate = False
ch = logging.StreamHandler()
if args.verbose:
    ch.setLevel(logging.DEBUG)
else:
    ch.setLevel(logging.INFO)

# Create formatter and add to ch
formatter = logging.Formatter('%(asctime)s :: %(levelname)s :: %(name)s, %(funcName)s :: line %(lineno)d :: %(message)s')
ch.setFormatter(formatter)

# Add ch to logger
logger.addHandler(ch)

# if args.verbose:
#     logging.getLogger().setLevel(logging.DEBUG)
# else:
#     logging.getLogger().setLevel(logging.INFO)

def normalize_path(in_path):
    # A quick function to ensure that any input paths are properly referenced
    return os.path.normpath(os.path.realpath(os.path.expanduser(in_path)))

def save_fig(filename, types=['png']):
    """
    Allows user to save figures as multiple file types at once
    Credit: https://stackoverflow.com/questions/17279651/save-a-figure-with-multiple-extensions
    """
    fig = plt.gcf()
    for filetype in types:
        logger.debug(f'Figure file type: {filetype}')
        logger.debug(f'Saving figure as {normalize_path(filename)}.{filetype}')
        fig.savefig(f'{normalize_path(filename)}.{filetype}', dpi=300, bbox_inches='tight')

class CRICKETS:

    def __init__(self, file_loc, n_divs, threshold):
        self.file = args.input_file
        self.n_divs = args.ndivs
        self.threshold = args.threshold

    def intro(self):
        """This function will initialize the Blimpy Waterfall object and 
            generate the info table
        """
        
        t_intro = time.time()
        logger.info('Generating waterfall object...')
        logger.info(f'Reading {self.file}...')

        ml = calcload.calc_max_load(self.file)
        wf = Waterfall(os.path.normpath(self.file), max_load = ml)
        logger.info(f'Done.')

        # Grab the name of the file for later use
        global file_name
        file_name = self.file[self.file.rfind('/')+1:self.file.rfind('.')]

        # Get power and frequency in increasing order
        logger.debug('Getting power and frequency in increasing order...')
        if wf.header['foff'] < 0:
            pows = np.flip(wf.data)
            freqs = wf.get_freqs()[::-1]
        else:
            pows = wf.data
            freqs = wf.get_freqs()
        logger.debug('Done.')

        # So:
        # pows_flipped is all of the powers in increasing order,
        # freqs_flipped is all of the frequencies in increasing order

        # Time-average the power
        logger.debug('Time-averaging power...')
        pows_mean = np.mean(pows, axis=0)[0]
        logger.debug('Done.')

        # Create table with time-averaged power and frequencies
        logger.debug('Creating info table...')
        global info_table 
        info_table = pd.DataFrame(columns=['freq', 'tavg_power'])
        info_table['freq'] = freqs
        info_table['tavg_power'] = pows_mean

        logger.debug(f'Done.')

        """This part of the function will calculate the excess kurtosis of each
        frequency bin
        """
        freqs = np.array(info_table['freq'])
        pows = np.array(info_table['tavg_power'])

        # Split frequency and time-averaged power into n_divs channels
        logger.debug(f'Splitting freq and tavg_power into bins.')
        freqs_binned = np.array_split(freqs, self.n_divs)
        pows = np.array_split(pows, self.n_divs)
        logger.debug(f'Done.')

        # Get excess kurtosis of all channels
        exkurts_list = []

        # Rescaling data so that excess kurtosis != inf ever (hopefully)
        logger.debug(f'Rescaling data.')
        for division in pows:
            exkurts_list.append(kurtosis(division/(10**9)))
        logger.debug(f'Done.')

        global exkurts
        exkurts = np.array(exkurts_list, dtype=np.float64)
        # print(f'\n\nHere is the np.abs(exkurts): {np.abs(exkurts)}\n\n')

        """Binning frequencies such that the labeled frequency
        is the bottom of the bin.
        i.e., if chnl[0] is 2010 MHz and each channel is 1 MHz, 
        then the bin from 2010 MHz to 2011 MHz will have a value of '2010'
        """
        logger.debug(f'Ensuring proper order of bins.')

        global bins
        bins = []
        for chnl in freqs_binned:
            bins.append(chnl[0])
        logger.debug(f'Done.')

        ##### Section 2 #####
        # This part of the function flags bins with high excess kurtosis.
        
        # masked_kurts is an array that has all channels with |excess kurtosis| > threshold masked out
        logger.debug(f'Getting mask for dirty bins.')
        global masked_kurts
        masked_kurts = ma.masked_where(np.abs(exkurts) > self.threshold, exkurts)
        bin_mask = ma.getmask(masked_kurts)
        logger.debug(f'Done.')

        """flagged_bins is an array that has the frequencies of the channels
        with excess kurtosis > threshold NOT masked out

        flagged_kurts masks the opposite elements as masked_kurts
        (i.e., it contains all of the kurtoses of the high RFI channels)
        """
        logger.debug(f'Making array with only dirty bins.')
        global flagged_bins
        global flagged_kurts
        
        flagged_bins = ma.masked_array(bins, mask=~bin_mask)
        
        flagged_kurts = ma.masked_array(exkurts, mask=~bin_mask)
        logger.debug(f'Done.')
        
        # TODO: Write a quick catch to make sure that if 0 bins were flagged then it just stops :)

        logger.info(f'{ma.count(flagged_bins)} out of {self.n_divs} channels flagged as having substantial RFI')

        # Bin width (for the files I am testing this with):
        # There are ~8 Hz in between each entry of freqs, and 32 MHz total
        # Given a total of 4194304 elements in freqs, if there are 256 bins, then each bin spans 16384 elements
        # If each bin spans 16384 elements, then it spans 125 kHz (125 kHz * 256 = 32 MHz)
        
        # Grab the bin width in terms of MHz (for convenience if needed in the future)
        logger.debug(f'Finding bin width in terms of f.')
        full_freq_range = np.array(info_table['freq'])[-1] - np.array(info_table['freq'])[0]
        # logger.info(f'Calculating exkurt of each bin.')
        global bin_width
        bin_width = full_freq_range / self.n_divs
        # logger.info(f'Calculating exkurt of each bin.')
        logger.debug(f'Done.')
        
        logger.debug(f'Full frequency range: {full_freq_range}')
        logger.debug(f'Bin width in terms of f: {bin_width}')

        logger.info(f'Exporting flagged bins to csv...')        
        
        # Grab the bin width in terms of the number of elements per bin
        # bin_width_elements = int(np.floor(len(freqs) / self.n_divs))
        
        # masked_freqs = ma.masked_array(freqs)

        # TODO: We just want the high RFI bins to be output, so this can be deleted? :O
        # logger.info(f'Creating array with all of the high RFI bins being masked.')
        # for rfi_bin in flagged_bins:
        #     try:
        #         # Get the frequency indices of the masked frequency bins and put them in a list
        #         xmin = np.where(freqs == rfi_bin)[0][0]
        #         xmax = xmin + bin_width_elements
        #         masking_indices = np.arange(xmin, xmax)
                
        #         # Create a masked array that masks the indices of the high RFI bins
        #         masked_freqs[masking_indices] = ma.masked
        #         freq_mask = ma.getmask(masked_freqs)
        #     except:
        #         pass

        # Export the flagged bins to a csv file
        # Get bin tops
        bin_tops = ma.masked_array([])
        for bin_bot in flagged_bins:
            bin_tops = ma.append(bin_tops, bin_bot + bin_width)

        # Format flagged_bins into a regular (non-masked) numpy array
        flagged_bins = ma.filled(flagged_bins, fill_value=np.NaN)

        # Turns the numpy arrays into pandas dataframes so they can be concatenated and exported
        export_bin_bots = pd.DataFrame(data=flagged_bins, columns=['rfi_bin_bots'])
        export_bin_tops = pd.DataFrame(data=bin_tops, columns=['rfi_bin_tops'])
        export_bin_exkurt = pd.DataFrame(data=flagged_kurts, columns=['exkurt'])

        # Concatenate dataframes
        export_concat = pd.concat([export_bin_exkurt,
                                export_bin_bots,
                                export_bin_tops], axis=1)

        # Sort dataframe by frequency, remove pandas indexing, remove blank lines
        export_df = export_concat.sort_values(by=['rfi_bin_bots']).reset_index(drop=True).dropna(how='all')

        # Write dataframe to csv at export_path
        if os.path.isdir(args.output_file):
           export_df.to_csv(f'{args.output_file}/crickets_{file_name}_{self.n_divs}_{self.threshold}.csv', index=False)
           logger.info(f'Exported flagged bins to {args.output_file}/crickets_{file_name}_{self.n_divs}_{self.threshold}.csv')
        elif os.path.isfile(args.output_file):
            export_df.to_csv(args.output_file, index=False)
            logger.info(f'Exported flagged bins to {args.output_file}')

        t_final_intro = time.time()
        logger.info(f'Done. Time elapsed for analysis: {t_final_intro - t_intro}\n\n')

    def plot_tavg_pwr(self, output_dest='', output_type=['png'], show_filtered_bins=True):
        """Plot the time-averaged power spectrum for a given Blimpy Waterfall object
        Inputs:
            output_dest: Location (including filename) to save output file
            output_type: Filetype of output
        """

        t_taps = time.time()
        logger.info("Plotting time-averaged power spectrum...")
        # Get frequencies and powers from info_table    
        # logger.warning(f'info_table: {type(info_table)}, {info_table}')

        freqs = np.array(info_table['freq'])
        pows = np.array(info_table['tavg_power'])

        # log_pows = np.log10(pows)

        # Plot time-averaged power
        fig, ax = plt.subplots()
        
        ax.set_xlim(np.amin(freqs), np.amax(freqs))

        # In case you want to change the frequency range of the plot:
        # ax.set_xlim(np.amin(freqs), 2270)

        """I was trying to put this on a log-axis, but it didn't work out
        It kept telling giving me:
        ValueError: Axis limits cannot be NaN or Inf
        
        Not sure why the limits are NaN or Inf, but I'll just leave it
        as non-log for now. This might actually have to do with the file I was
        using:

        TCOS0001_sb43905589_1_1_001.60074.91866136574.3.1.AC.C224-8Hz-beam0001.fil
        """
        ax.set_ylim(np.amin(pows), np.amax(pows))
        # ax.set_yscale('log')
        
        ax.set_xlabel('Frequency (MHz)')
        ax.set_ylabel('log Time-Averaged Power (W)')
        ax.set_title(f'Time-Averaged Power Spectrum of\n{file_name} (n_divs={self.n_divs}, threshold={self.threshold})', y=1.06)

        ax.plot(freqs, pows,
                label='Time-averaged power spectrum',
                c='#1f1f1f')


        # Plot frequency bins that were flagged as RFI
        if show_filtered_bins == True:
            full_freq_range = freqs[-1] - freqs[0]
            logger.debug(f'full_freq_range: {full_freq_range}')

            for rfi_bin in flagged_bins:
                xmin = rfi_bin
                xmax = rfi_bin + bin_width
                flagged_line = plt.axvspan(xmin=xmin, xmax=xmax, ymin=0, ymax=1, color='red', alpha=0.5)

            flagged_line.set_label('Dirty channels')
            ax.legend(fancybox=True, shadow=True, loc='lower center', bbox_to_anchor=(0.5, 0.91), ncols=1)
        else:
            ax.legend(fancybox=True, shadow=True, loc='lower center', bbox_to_anchor=(0.5, 0.91), ncols=1)
        
        logger.debug(f'tavg_power spectrum output folder: {output_dest}')
        save_fig(os.path.join(normalize_path(output_dest), f'plot_tavg_power_{file_name}_{self.n_divs}_{self.threshold}'), types=output_type)

        for filetype in output_type:
            logger.info(f"tavg_power plot ({filetype}) generated at {os.path.join(normalize_path(output_dest), f'plot_tavg_power_{file_name}_{self.n_divs}_{self.threshold}.{filetype}')}")
        t_final_taps = time.time()
        logger.info(f'Time elapsed for plotting time-averaged power spectrum: {t_final_taps - t_taps}\n\n')

    def plot_exkurt(self, unfiltered=False, clean_chnls=True, rfi=True,
                    output_dest='', output_type=['png']):
        """ This function plots the excess kurtosis of each frequency channel for a
        specified waterfall object.
        
        Inputs:
            unfiltered: If true, plot the data before RFI filtering has occurred
            clean_chnls: If true, plot the data after RFI has been filtered out
            rfi: If true, plot the channels that have been marked as RFI
            output_dest: Location (including filename) to save output file
            output_type: File type(s) of output
        """


        t_exkurt = time.time()
        logger.info("Plotting excess kurtosis vs. frequency...")
        
        freqs = np.array(info_table['freq'])
        pows = np.array(info_table['tavg_power'])

        # Create the plot
        fig, ax = plt.subplots()

        # plt.figure(figsize=(100, 100))
        ax.set_xlabel('Frequency (MHz)')
        ax.set_ylabel('Excess Kurtosis')
        ax.set_title(f"Excess Kurtosis of\n{file_name} (n_divs={self.n_divs}, threshold={self.threshold})", y=1.06)

        # Grab the bin width in terms of the number of elements per bin
        bin_width_elements = int(np.floor(len(freqs) / self.n_divs))

        logger.debug(f'Creating array with all of the high RFI bins being masked.')
        masked_freqs = ma.masked_array(freqs)
        for rfi_bin in flagged_bins:
            try:
                # Get the frequency indices of the masked frequency bins and put them in a list
                xmin = np.where(freqs == rfi_bin)[0][0]
                xmax = xmin + bin_width_elements
                masking_indices = np.arange(xmin, xmax)

                # Create a masked array that masks the indices of the high RFI bins
                masked_freqs[masking_indices] = ma.masked
                freq_mask = ma.getmask(masked_freqs)
            except:
                pass
        logger.debug(f'Done.')

        # Plot all data
        logger.debug("Plotting all data...")
        if unfiltered:
            ax.plot(bins, exkurts, 'o', c='black', label='Unfiltered data') # Color is a nice black
        logger.debug("Done...")

        # Plot the low RFI channels
        logger.debug("Plotting clean channels...")
        if clean_chnls:
            ax.plot(bins, masked_kurts.filled(np.nan), '.', c='#43cc5c', label='Clean channels') # Color is a nice green
        logger.debug("Done...")

        # Plot the high RFI channels
        logger.debug("Plotting dirty channels...")
        if rfi:
            ax.plot(flagged_bins, flagged_kurts, '.', c='red', label='Dirty channels') # Color is a nice red
        logger.debug("Done...")

        ax.set_xlim(np.amin(freqs), np.amax(freqs))

        # logger.debug("Setting excess kurtosis y-axis limits...")
        # ax.set_ylim(np.amin(exkurts), np.amax(exkurts))
        # logger.debug("Done...")

        ax.legend(fancybox=True,shadow=True, loc='lower center', bbox_to_anchor=(0.5, 0.95), ncols=3)

        logger.debug(f'exkurt plot output folder: {output_dest}')
        save_fig(os.path.join(normalize_path(output_dest), f'plot_exkurt_{file_name}_{self.n_divs}_{self.threshold}'), types=output_type)

        for filetype in output_type:
            logger.info(f"exkurt plot ({filetype}) generated at {os.path.join(normalize_path(output_dest), f'plot_exkurt_{file_name}_{self.n_divs}_{self.threshold}.{filetype}')}")

        t_final_exkurt = time.time()
        logger.info(f"Time elapsed for plotting excess kurtosis: {t_final_exkurt - t_exkurt}")


if __name__ == "__main__":
    t_start = time.time()
    file_path = normalize_path(args.input_file)

    cricket = CRICKETS(args.input_file, args.ndivs, args.threshold)
    cricket.intro()
    
    if args.plot:
        cricket.plot_tavg_pwr(output_dest = args.plot,
                              output_type = args.plot_file_types,
                              show_filtered_bins = True)
        cricket.plot_exkurt(unfiltered = False, clean_chnls = True, rfi = True,
                            output_dest = args.plot,
                            output_type = args.plot_file_types)

    t_end = time.time()
    logger.info(f'Total time elapsed (analysis and plotting): {t_end - t_start}')