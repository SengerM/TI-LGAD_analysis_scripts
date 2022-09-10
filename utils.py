import datetime
import time
from scipy.stats import median_abs_deviation
from the_bureaucrat.bureaucrats import RunBureaucrat # https://github.com/SengerM/the_bureaucrat
import pandas
from huge_dataframe.SQLiteDataFrame import load_whole_dataframe # https://github.com/SengerM/huge_dataframe
from pathlib import Path
import numpy
import warnings

def create_a_timestamp():
	time.sleep(1) # This is to ensure that no two timestamps are the same.
	return datetime.datetime.now().strftime("%Y%m%d%H%M%S")

def integrate_distance_given_path(points:list):
	"""Given a list of points of the form `[(x0,y0,z0),(x1,y1,z1),...]`
	it calculates the distance at each point, starting from 0 and assuming
	linear interpolation."""
	def calculate_distance(p1, p2):
		return sum([(x1-x2)**2 for x1,x2 in zip(p1,p2)])**.5
	
	distance = [0]
	for i,p in enumerate(points):
		if i == 0:
			continue
		distance.append(distance[-1] + calculate_distance(points[i-1],p))
	return distance

def generate_distance_vs_n_position(df)->pandas.Series:
	"""Given a `df` that posses the columns x,y,z and n_position, integrates
	the distance and returns a series with this information.
	
	Example input:
	```
											   x (m)     y (m)     z (m)  ...
	n_position n_trigger n_channel n_pulse                                ...
	0          0         1         1       -0.002926  0.001314  0.071339  ...
								   2       -0.002926  0.001314  0.071339  ...
						 4         1       -0.002926  0.001314  0.071339  ...
								   2       -0.002926  0.001314  0.071339  ...
			   1         1         1       -0.002926  0.001314  0.071339  ...
	...                                          ...       ...       ...  ...
	333        109       4         2       -0.002926  0.001647  0.071339  ...
			   110       1         1       -0.002926  0.001647  0.071339  ...
								   2       -0.002926  0.001647  0.071339  ...
						 4         1       -0.002926  0.001647  0.071339  ...
								   2       -0.002926  0.001647  0.071339  ...

	[148296 rows x 35 columns]
	```
	Example output:
	```
	n_position
	0      0.000000e+00
	1      9.960938e-07
	2      2.001953e-06
	3      2.998047e-06
	4      4.003906e-06
			   ...     
	329    3.289941e-04
	330    3.300000e-04
	331    3.309961e-04
	332    3.320020e-04
	333    3.329980e-04
	Name: Distance (m), Length: 334, dtype: float64
	```
	"""
	COORDINATES_COLUMNS = ['x (m)', 'y (m)', 'z (m)']
	if 'n_position' not in df.index.names:
		raise ValueError(f'`n_position` must be one index level of `df`.')
	if any([col not in df.columns for col in COORDINATES_COLUMNS]):
		raise ValueError(f'Column(s) {set(COORDINATES_COLUMNS)-set(df.columns)} not present in `df`.')
	df = df.reset_index(drop=False)
	df = df.set_index('n_position')
	df = df[~df.index.duplicated(keep='first')]
	df2 = pandas.DataFrame(index=df.index)
	df2['Distance (m)'] = integrate_distance_given_path(list(df[COORDINATES_COLUMNS].to_numpy()))
	distance = df2.droplevel(list(set(df2.index.names)-{'n_position'}),axis=0)['Distance (m)']
	return distance

def kMAD(x,nan_policy='omit'):
	"""Calculates the median absolute deviation multiplied by 1.4826... 
	which should converge to the standard deviation for Gaussian distributions,
	but is much more robust to outliers than the std."""
	k_MAD_TO_STD = 1.4826 # https://en.wikipedia.org/wiki/Median_absolute_deviation#Relation_to_standard_deviation
	return k_MAD_TO_STD*median_abs_deviation(x,nan_policy=nan_policy)

def interlace(lst):
	# https://en.wikipedia.org/wiki/Interlacing_(bitmaps)
	lst = sorted(lst)[::-1]
	result = [lst[0], lst[-1]]
	ranges = [(1, len(lst) - 1)]
	for start, stop in ranges:
		if start < stop:
			middle = (start + stop) // 2
			result.append(lst[middle])
			ranges += (start, middle), (middle + 1, stop)
	return result

def load_parsed_from_waveforms_and_measured_data_in_TCT_1D_scan(bureaucrat:RunBureaucrat)->pandas.DataFrame:
	"""If `bureaucrat` points to a run with a "TCT_1D_scan", this function
	returns a dataframe with the data parsed from the scan and the correct
	index.
	
	Example
	-------
	```
											   x (m)     y (m)     z (m)  ...      t_70 (s)      t_80 (s)      t_90 (s)
	n_position n_trigger n_channel n_pulse                                ...                                          
	0          0         1         1       -0.002926  0.001314  0.071339  ...  1.442067e-07  1.442211e-07  1.442356e-07
								   2       -0.002926  0.001314  0.071339  ...  2.586039e-07  2.586193e-07  2.586346e-07
						 4         1       -0.002926  0.001314  0.071339  ...  2.160344e-07  2.160563e-07  2.160781e-07
								   2       -0.002926  0.001314  0.071339  ...  3.039999e-07  3.040166e-07  3.040333e-07
			   1         1         1       -0.002926  0.001314  0.071339  ...  1.605778e-07  1.605852e-07  1.605926e-07
	...                                          ...       ...       ...  ...           ...           ...           ...
	333        109       4         2       -0.002926  0.001647  0.071339  ...  2.409688e-07  2.409792e-07  2.409896e-07
			   110       1         1       -0.002926  0.001647  0.071339  ...  2.122895e-07  2.123097e-07  2.123298e-07
								   2       -0.002926  0.001647  0.071339  ...  2.602939e-07  2.603086e-07  2.603293e-07
						 4         1       -0.002926  0.001647  0.071339  ...  2.155185e-07  2.155290e-07  2.155395e-07
								   2       -0.002926  0.001647  0.071339  ...  2.603506e-07  2.603671e-07  2.603835e-07

	[148296 rows x 35 columns]
	```
	"""
	Joaquin = bureaucrat
	Joaquin.check_these_tasks_were_run_successfully(['TCT_1D_scan','parse_waveforms'])
	parsed_from_waveforms_df = load_whole_dataframe(Joaquin.path_to_directory_of_task('parse_waveforms')/'parsed_from_waveforms.sqlite')
	measured_data_df = load_whole_dataframe(Joaquin.path_to_directory_of_task('TCT_1D_scan')/'measured_data.sqlite')
	return measured_data_df.merge(parsed_from_waveforms_df, left_index=True, right_index=True).droplevel('n_waveform',axis=0)

def tag_channels_left_right(bureaucrat:RunBureaucrat):
	Lars = bureaucrat
	Lars.check_these_tasks_were_run_successfully(['TCT_1D_scan','parse_waveforms'])
	
	with Lars.handle_task('tag_channels_left_right') as Lars_employee:
		df = load_parsed_from_waveforms_and_measured_data_in_TCT_1D_scan(Lars)
		left_data = df.query(f'n_position<{numpy.mean(df.index.get_level_values("n_position"))}')
		right_data = df.query(f'n_position>{numpy.mean(df.index.get_level_values("n_position"))}')
		n_channels = set(df.index.get_level_values('n_channel'))
		for n_channel in n_channels:
			if left_data.query(f'n_channel=={n_channel}')['Amplitude (V)'].mean(skipna=True) > left_data.query(f'n_channel!={n_channel}')['Amplitude (V)'].mean(skipna=True):
				mapping = {'left': n_channel, 'right': list(n_channels-{n_channel})[0]}
			else:
				mapping = {'right': n_channel, 'left': list(n_channels-{n_channel})[0]}
		tags_df = pandas.DataFrame({'n_channel': [mapping[t] for t in sorted(mapping)], 'channel_position': sorted(mapping)})
		tags_df.set_index('n_channel',inplace=True)
		tags = tags_df['channel_position'] # Convert into series
		print(tags)
		tags.to_pickle(Lars_employee.path_to_directory_of_my_task/'channels_position.pickle')

if __name__=='__main__':
	tag_channel_position(RunBureaucrat(Path('/home/alf/cernbox/projects/4D_sensors/TI-LGAD_FBK_RD50_1/measurements_data/220715_second_campaign/20220906_testing_the_setup/TCT_scans/subruns/20220908134101_hate_the_bugged_oscilloscope/TCT_1D_scan_sweeping_bias_voltage/subruns/20220908134101_hate_the_bugged_oscilloscope_220V')))
	
