import datetime
import time
from scipy.stats import median_abs_deviation
from the_bureaucrat.bureaucrats import RunBureaucrat # https://github.com/SengerM/the_bureaucrat
import pandas
from huge_dataframe.SQLiteDataFrame import load_whole_dataframe # https://github.com/SengerM/huge_dataframe
from pathlib import Path
import numpy
import warnings
import grafica.plotly_utils.utils as graficas_px_utils

graficas_px_utils.set_my_template_as_default()

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

def tag_channels_left_right(bureaucrat:RunBureaucrat, force:bool=False):
	"""Tags each of the two channel numbers as `"left"` or `"right"`."""
	Lars = bureaucrat
	Lars.check_these_tasks_were_run_successfully(['TCT_1D_scan','parse_waveforms'])
	
	if force==False and Lars.was_task_run_successfully('tag_channels_left_right'):
		return
	
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
		tags.to_pickle(Lars_employee.path_to_directory_of_my_task/'channels_position.pickle')

def resample_by_events(data_df):
	N_EVENT_COLUMN_IDENTIFIERS = ['n_position','n_trigger']
	columns = [_ for _ in data_df.index.names if _ not in N_EVENT_COLUMN_IDENTIFIERS]
	initial_dtypes = data_df.dtypes
	resampled_df = data_df.reset_index(drop=False).pivot(
		index = N_EVENT_COLUMN_IDENTIFIERS,
		columns = columns,
		values = set(data_df.columns),
	)
	resampled_df = resampled_df.groupby('n_position').sample(frac=1, replace=True)
	resampled_df = resampled_df.stack(level=columns)
	if any([initial_dtypes[_]!=resampled_df.dtypes[_] for _ in data_df.columns]):
		warnings.warn(f'There was an automatic cast by Pandas while running `pivot` to resample, please check this because it may not broke the code but severely affect performance after resampling.')
	return resampled_df

def normalize_data(data_df, normalization_factors_df):
	"""Given normalization factors in the form
	```
					  Amplitude (V)                            
							 offset divide_by_this_to_normalize
	n_channel n_pulse                                          
	1         1            0.013011                    0.327003
			  2            0.012813                    0.240779
	4         1            0.013397                    0.343928
			  2            0.013309                    0.253097
	```
	and data e.g.
	```
											Amplitude (V) normalized
	n_position n_trigger n_channel n_pulse                          
	0          0         1         1                        0.012532
								   2                        0.013547
						 4         1                        0.017058
								   2                        0.015573
			   1         1         1                        0.012446
	...                                                          ...
	333        109       4         2                        0.012202
			   110       1         1                        0.012348
								   2                        0.012273
						 4         1                        0.012907
								   2                        0.012770

	[148296 rows x 1 columns]
	```
	this function returns a new data frame with the columns present in 
	`normalization_factors_df` (in this example `"Amplitude (V)"`) but
	normalized.
	"""
	normalized = pandas.DataFrame(index=data_df.index)
	for col in set(normalization_factors_df.columns.get_level_values(0)):
		normalized[f'{col} normalized'] = data_df[col]
	normalized.reset_index(inplace=True, drop=False)
	normalized.set_index(normalization_factors_df.index.names, inplace=True)
	normalized.sort_index(inplace=True)
	for col in set(normalization_factors_df.columns.get_level_values(0)):
		for i in normalization_factors_df.index:
			normalized.loc[i,f'{col} normalized'] -= normalization_factors_df.loc[i,(col,'offset')]
			normalized.loc[i,f'{col} normalized'] /= normalization_factors_df.loc[i,(col,'divide_by_this_to_normalize')]
	normalized.reset_index(inplace=True, drop=False)
	normalized.set_index(data_df.index.names, inplace=True)
	return normalized

def normalization_factors_for_amplitude(bureaucrat:RunBureaucrat, approximate_window_size_meters:float, approximate_laser_size_meters:float, force:bool=False):
	"""Calculates the normalization factors for the amplitude and charge
	of each channel such that they go from 0 to 1."""
	bureaucrat.check_these_tasks_were_run_successfully(['TCT_1D_scan','parse_waveforms','tag_channels_left_right'])
	
	if force==False and bureaucrat.was_task_run_successfully('normalization_factors_for_amplitude'):
		return
	
	with bureaucrat.handle_task('normalization_factors_for_amplitude') as employee:
		data_df = load_parsed_from_waveforms_and_measured_data_in_TCT_1D_scan(bureaucrat)
		channel_positions = pandas.read_pickle(bureaucrat.path_to_directory_of_task('tag_channels_left_right')/'channels_position.pickle')
		distance = generate_distance_vs_n_position(data_df)
		
		data_df = data_df[['Amplitude (V)']]
		data_df = data_df.join(distance, on='n_position')
		
		distance_left_window_ends = distance.mean()-approximate_laser_size_meters
		distance_left_window_begins = distance.mean()-approximate_window_size_meters/2+approximate_laser_size_meters
		distance_right_wndow_begins = distance.mean()+approximate_laser_size_meters
		distance_right_window_ends = distance.mean()+approximate_window_size_meters/2-approximate_laser_size_meters
		
		normalization_data = []
		for n_channel in set(data_df.index.get_level_values('n_channel')):
			for n_pulse in set(data_df.index.get_level_values('n_pulse')):
				use_this_data = data_df.query(f'n_channel=={n_channel}').query(f'n_pulse=={n_pulse}')
				if channel_positions.loc[n_channel] == 'left':
					data_where_I_expect_amplitude_to_be_maximum = use_this_data.query(f'`Distance (m)`<{distance_left_window_ends}').query(f'`Distance (m)`>{distance_left_window_begins}')
					data_where_I_expect_amplitude_to_be_minimum = use_this_data.query(f'`Distance (m)`>{distance_right_wndow_begins}')
				else: # right channel
					data_where_I_expect_amplitude_to_be_maximum = use_this_data.query(f'`Distance (m)`>{distance_right_wndow_begins}').query(f'`Distance (m)`<{distance_right_window_ends}')
					data_where_I_expect_amplitude_to_be_minimum = use_this_data.query(f'`Distance (m)`<{distance_left_window_ends}')
				amplitude_offset = data_where_I_expect_amplitude_to_be_minimum['Amplitude (V)'].median()
				amplitude_divide_by_this_to_normalize = (data_where_I_expect_amplitude_to_be_maximum['Amplitude (V)']-amplitude_offset).median()
				df = pandas.DataFrame({'n_channel':n_channel}, index=[0]).set_index('n_channel')
				df = pandas.DataFrame(
					{
						'Amplitude (V) offset': amplitude_offset,
						'Amplitude (V) divide_by_this_to_normalize': amplitude_divide_by_this_to_normalize,
						'n_channel': n_channel,
						'n_pulse': n_pulse,
					},
					index = [0],
				).set_index(['n_channel','n_pulse'])
				df.columns = pandas.MultiIndex.from_product([['Amplitude (V)'],['offset','divide_by_this_to_normalize']])
				normalization_data.append(df)
		normalization_data = pandas.concat(normalization_data)
		
		normalization_data.to_pickle(employee.path_to_directory_of_my_task/'normalization_data.pickle')
		
		averaged_in_position_df = normalize_data(data_df, normalization_data).groupby(['n_position','n_channel','n_pulse']).agg([numpy.nanmedian, kMAD])
		averaged_in_position_df.columns = [f'{col[0]} {col[1]}' for col in averaged_in_position_df.columns]
		averaged_in_position_df = averaged_in_position_df.join(distance, on='n_position')
		averaged_in_position_df = averaged_in_position_df.join(channel_positions, on='n_channel')
		
		fig = graficas_px_utils.line(
			data_frame = averaged_in_position_df.reset_index().sort_values(['channel_position','n_position']),
			x = 'Distance (m)',
			y = 'Amplitude (V) normalized nanmedian',
			error_y = 'Amplitude (V) normalized kMAD',
			error_y_mode = 'bands',
			color = 'channel_position',
			line_dash = 'n_pulse',
			title = f'Normalized amplitude<br><sup>Run: {bureaucrat.run_name}</sup>',
		)
		fig.write_html(
			employee.path_to_directory_of_my_task/'amplitude normalized.html',
			include_plotlyjs = 'cdn',
		)

if __name__=='__main__':
	import argparse

	parser = argparse.ArgumentParser()
	parser.add_argument('--dir',
		metavar = 'path', 
		help = 'Path to the base measurement directory.',
		required = True,
		dest = 'directory',
		type = str,
	)
	parser.add_argument(
		'--force',
		help = 'If this flag is passed, it will force the calculation even if it was already done beforehand. Old data will be deleted.',
		required = False,
		dest = 'force',
		action = 'store_true'
	)
	args = parser.parse_args()
	
	Enrique = RunBureaucrat(Path(args.directory))
	normalization_factors_for_amplitude(
		Enrique,
		approximate_window_size_meters = 250e-6,
		approximate_laser_size_meters = 11e-6,
	)
	
