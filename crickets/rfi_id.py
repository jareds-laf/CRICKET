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

# parser = argparse.ArgumentParser(
#                     description='Flag RFI heavy frequency channels based on the excess kurtosis of each channel.')

# # Input/analysis arguments
# parser.add_argument('--input_file',
# 		    help='(Required) Path to input filterbank file (including file name).', 
# 			type=str,
# 			required=True)
# parser.add_argument('--output_file',
# 		    help='(Required) Path to output csv file (optionally including file name).',
# 			type=str,
# 			required=True)
# parser.add_argument('--threshold', '-t',
# 		    help='(Required) Minimum value of excess kurtosis used to flag channels with significant RFI. Can be any decimal number.',
# 			type=float,
# 			default=5,
# 			required=True)
# parser.add_argument('--ndivs', '-n',
# 		    help='(Required) Number of frequency bins to split waterfall object into. Can be any integer.',
# 			type=int,
# 		    default=256,
# 			required=True)

# Plotting arguments
# TODO: Better implementation of the plotting arguments!
# parser.add_argument('--plot', '-p',
# 		    help='(Optional) Choose whether or not to generate time-averaged power spectrum and excess kurtosis vs. frequency plots. Give output path for plots here (NOT including file name).',
# 			# action='store_true',
# 			type=str,
# 			required=False)
# parser.add_argument('--plot_file_types', '--pft',
# 			help='(Optional, unless -p is given) Output file types (can be pdf, png, and/or jpg). Specify as many of these as you want!',
# 			choices=['png', 'pdf', 'jpg'],
# 			nargs='+',
# 			required=False)
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
# parser.add_argument('--verbose', '-v',
# 		    help='(Optional) Print more information about the input variables and the processes currently running.',
# 			action='store_true',
# 			required=False)

# args = parser.parse_args()

# Set up logger
logger = logging.getLogger('analysis')

logger.propagate = False
ch = logging.StreamHandler()

# Create formatter
formatter = logging.Formatter('%(asctime)s :: %(levelname)s :: %(name)s, %(funcName)s :: line %(lineno)d :: %(message)s')

# Add formatter to ch
ch.setFormatter(formatter)

# Add ch to logger
logger.addHandler(ch)

def normalize_path(in_path):
    # A quick function to ensure that any input paths are properly referenced
    return os.path.normpath(os.path.realpath(os.path.expanduser(in_path)))

class RID:

    def __init__(self, file_loc, n_divs, threshold):
        self.file = file_loc
        self.n_divs = n_divs
        self.threshold = threshold

    def intro(self):
        """This function will initialize the Blimpy Waterfall object and 
            generate the info table
        """
        
        t0 = time.time()
        logger.info('Generating waterfall object...')
        logger.info(f'Reading {self.file}...')

        ml = calcload.calc_max_load(self.file)
        wf = Waterfall(os.path.normpath(self.file), max_load = ml)
        logger.info(f'Done.')

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
        freqs = np.array_split(freqs, self.n_divs)
        pows = np.array_split(pows, self.n_divs)
        logger.debug(f'Done.')

        # Get excess kurtosis of all channels
        exkurts_list = []

        # Rescaling data so that excess kurtosis != inf ever (hopefully)
        logger.debug(f'Rescaling data.')
        for division in pows:
            exkurts_list.append(kurtosis(division/(10**9)))
        logger.debug(f'Done.')

        exkurts = np.array(exkurts_list, dtype=np.float64)
        # print(f'\n\nHere is the np.abs(exkurts): {np.abs(exkurts)}\n\n')

        """Binning frequencies such that the labeled frequency
        is the bottom of the bin.
        i.e., if chnl[0] is 2010 MHz and each channel is 1 MHz, 
        then the bin from 2010 MHz to 2011 MHz will have a value of '2010'
        """
        logger.debug(f'Ensuring proper order of bins.')
        bins = []
        for chnl in freqs:
            bins.append(chnl[0])
        logger.debug(f'Done.')

        ##### Section 2 #####
        # This part of the function flags bins with high excess kurtosis.
        
        # masked_kurts is an array that has all channels with |excess kurtosis| > threshold masked out
        logger.debug(f'Getting mask for dirty bins.')
        masked_kurts = ma.masked_where(np.abs(exkurts) > self.threshold, exkurts)
        bin_mask = ma.getmask(masked_kurts)
        logger.debug(f'Done.')

        """flagged_bins is an array that has the frequencies of the channels
        with excess kurtosis > threshold NOT masked out

        flagged_kurts masks the opposite elements as masked_kurts
        (i.e., it contains all of the kurtoses of the high RFI channels)
        """
        logger.debug(f'Making array with only dirty bins.')
        flagged_bins = ma.masked_array(bins, mask=~bin_mask)
        flagged_kurts = ma.masked_array(exkurts, mask=~bin_mask)
        logger.debug(f'Done.')
        
        logger.info(f'{ma.count(flagged_bins)} out of {self.n_divs} channels flagged as having substantial RFI')

        # TODO: Make sure this isn't completely broken!

        # Bin width (for the files I am testing this with):
        # There are ~8 Hz in between each entry of freqs, and 32 MHz total
        # Given a total of 4194304 elements in freqs, if there are 256 bins, then each bin spans 16384 elements
        # If each bin spans 16384 elements, then it spans 125 kHz (125 kHz * 256 = 32 MHz)
        
        # Grab the bin width in terms of MHz (for convenience if needed in the future)
        logger.debug(f'Finding bin width in terms of f.')
        full_freq_range = np.array(info_table['freq'])[-1] - np.array(info_table['freq'])[0]
        # logger.info(f'Calculating exkurt of each bin.')
        bin_width = full_freq_range / self.n_divs
        # logger.info(f'Calculating exkurt of each bin.')
        logger.debug(f'Done.')
        
        logger.debug(f"Full frequency range: {full_freq_range}")
        logger.debug(f"Bin width in terms of f: {bin_width}")

        logger.info(f'Exporting flagged bins to csv...')        
        # Grab the bin width in terms of the number of elements per bin
        bin_width_elements = int(np.floor(len(freqs) / self.n_divs))
        
        masked_freqs = ma.masked_array(freqs)

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
        # TODO: Make sure the output path follows the export path specified by parser
        export_df.to_csv("/home/jsofair/misc_testing/oop_crickets_output.csv", index=False)

        t_final = time.time()
        logger.info(f'Done. Total time elapsed: {t_final - t0}')

    def plot_tavg_pwr(self):
        """This function will plot the time-averaged power spectrum
        """
        # TODO: On hold until I figure out how to get the plots working in the vsc interactive window again :(

        info_table = self.intro()

        fig, ax = plt.subplots()
        ax.plot(info_table['freq'], info_table['tavg_power'])
        plt.show()


if __name__ == "__main__":
    wf_path = normalize_path('/mnt/cosmic-storage-1/data0/jsofair/fil_60288_83463_1613984375_HD_4628_0001-beam0000.fbh5.fil')
    test = RID(wf_path, 256, 1)

    # print(wf_path)
    
    test.intro()
    # test.plot_tavg_pwr()

    # fig, ax = plt.subplots()
    # ax.plot(np.linspace(0,10), np.linspace(10,20))
    # plt.show()