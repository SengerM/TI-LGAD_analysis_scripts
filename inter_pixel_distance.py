from the_bureaucrat.bureaucrats import RunBureaucrat # https://github.com/SengerM/the_bureaucrat
from pathlib import Path
import numpy
import pandas
import warnings
from scipy import interpolate
import utils
import grafica.plotly_utils.utils as graficas_px_utils
import plotly.graph_objects as go
import plotly.express as px
from huge_dataframe.SQLiteDataFrame import SQLiteDataFrameDumper, load_whole_dataframe # https://github.com/SengerM/huge_dataframe

graficas_px_utils.set_my_template_as_default()

def calculate_inter_pixel_distance_by_linear_interpolation_using_normalized_amplitude(data_df:pandas.DataFrame, distance:pandas.Series, channel_positions:pandas.Series, approximate_window_size_meters:float, threshold_percent:int=50)->dict:
	"""Calculate the inter-pixel distance using linear interpolation in
	along the average normalized amplitude along the 1D scan.
	
	Arguments
	---------
	data_df: pandas.DataFrame
		A data frame with data from the scan, specifically the `'Amplitude (V) normalized'`.
		Example
		```
												Amplitude (V)  Amplitude (V) normalized
		n_position n_trigger n_channel n_pulse                                         
		0          0         1         1             0.012528                 -0.000237
									   2             0.011494                 -0.010244
							 4         1             0.016652                  0.019544
									   2             0.012556                 -0.009773
				   1         1         1             0.012229                 -0.002185
		...                                               ...                       ...
		333        109       4         2             0.014650                  0.009284
				   110       1         1             0.013633                  0.006955
									   2             0.010413                 -0.020351
							 4         1             0.013372                 -0.001813
									   2             0.010560                 -0.027941

		[148296 rows x 2 columns]
		```
	distance: pandas.Series
		A series containing the distance at each position. Example:
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
	channel_positions: pandas.Series
		A series telling how to map `n_channel` to left and right. Example:
		```
		n_channel
		4     left
		1    right
		Name: channel_position, dtype: object
		```
	approximate_window_size_meters: float
	threshold_percent: int, default 50
		The height at which to calculate the inter-pixel distance in the
		normalized amplitude. Typically you just want 50 %.
	
	Returns
	-------
	IPD_data: dict
		A dictionary with the results, e.g.
		```
		{'Inter-pixel distance (m)': 8.582276919577743e-06, 'Left pixel distance (m)': array(0.00016416), 'Right pixel distance (m)': array(0.00017275)}
		```
	"""
	REQUIRED_COLUMNS = {'Amplitude (V) normalized'}
	if not all([col in data_df.columns for col in REQUIRED_COLUMNS]):
		raise ValueError(f'`data_df` must contain the columns {REQUIRED_COLUMNS} but it is missing {set(REQUIRED_COLUMNS)-set(data_df.columns)}...')
	if distance.name != 'Distance (m)':
		raise ValueError(f'`distance`s name must be "Distance (m)".')
	
	data_df = data_df.query('n_pulse==1') # Use only the first pulse.
	
	averaged_in_position = data_df[['Amplitude (V) normalized']].groupby(['n_position','n_channel','n_pulse']).agg([numpy.nanmedian])
	averaged_in_position.columns = [f'{col[0]} {col[1]}' for col in averaged_in_position.columns]
	averaged_in_position = averaged_in_position.join(distance, on='n_position')
	averaged_in_position = averaged_in_position.join(channel_positions, on='n_channel')
	
	rows = (averaged_in_position['Distance (m)'] > averaged_in_position['Distance (m)'].mean() - approximate_window_size_meters/2)
	rows &= (averaged_in_position['Distance (m)'] < averaged_in_position['Distance (m)'].mean() + approximate_window_size_meters/2)
	averaged_in_position = averaged_in_position.loc[rows]
	
	threshold_distance_for_each_pad = {}
	for channel_position in {'left','right'}:
		df = averaged_in_position.loc[rows].query(f'channel_position=={repr(channel_position)}')
		distance_vs_charge_linear_interpolation = interpolate.interp1d(
			x = df['Amplitude (V) normalized nanmedian'],
			y = df['Distance (m)'],
		)
		threshold_distance_for_each_pad[channel_position] = distance_vs_charge_linear_interpolation(threshold_percent/100)
	return {
		'Inter-pixel distance (m)': threshold_distance_for_each_pad['right']-threshold_distance_for_each_pad['left'],
		'Left pixel distance (m)': threshold_distance_for_each_pad['left'],
		'Right pixel distance (m)': threshold_distance_for_each_pad['right'],
	}

def inter_pixel_distance(bureaucrat:RunBureaucrat, number_of_bootstrapped_replicas:int=0, threshold_percent:int=50, force:bool=False):
	bureaucrat.check_these_tasks_were_run_successfully(['TCT_1D_scan','parse_waveforms','normalization_factors_for_amplitude','tag_channels_left_right'])
	
	if force==False and bureaucrat.was_task_run_successfully('inter_pixel_distance'):
		return
	
	with bureaucrat.handle_task('inter_pixel_distance') as employee:
		parsed_from_waveforms = load_whole_dataframe(bureaucrat.path_to_directory_of_task('parse_waveforms')/'parsed_from_waveforms.sqlite')
		parsed_from_waveforms = parsed_from_waveforms[['Amplitude (V)']]
		
		measured_data = load_whole_dataframe(bureaucrat.path_to_directory_of_task('TCT_1D_scan')/'measured_data.sqlite')
		distance = utils.generate_distance_vs_n_position(measured_data)
		
		normalization_factors_df = pandas.read_pickle(bureaucrat.path_to_directory_of_task('normalization_factors_for_amplitude')/'normalization_data.pickle')
		
		channel_positions = pandas.read_pickle(bureaucrat.path_to_directory_of_task('tag_channels_left_right')/'channels_position.pickle')
		
		parsed_from_waveforms.index = measured_data.index.droplevel(level='n_waveform')
		parsed_from_waveforms = parsed_from_waveforms.join(utils.normalize_data(parsed_from_waveforms, normalization_factors_df))
		parsed_from_waveforms = parsed_from_waveforms.query('n_pulse==1')
		
		IPD_data = []
		for n_bootstrap in range(number_of_bootstrapped_replicas+1):
			if n_bootstrap == 0:
				df = parsed_from_waveforms.copy()
			else:
				df = utils.resample_by_events(parsed_from_waveforms)
			_ = calculate_inter_pixel_distance_by_linear_interpolation_using_normalized_amplitude(
				data_df = df,
				approximate_window_size_meters = 250e-6,
				channel_positions = channel_positions,
				distance = distance,
				threshold_percent = threshold_percent,
			)
			_['n_bootstrap'] = n_bootstrap
			_ = pandas.DataFrame(_, index=[0]).set_index('n_bootstrap')
			IPD_data.append(_)
		IPD_data = pandas.concat(IPD_data)
		IPD_data = IPD_data.agg([numpy.nanmean, numpy.nanstd])
		
		IPD_data.to_pickle(employee.path_to_directory_of_my_task/'IPD_data.pickle')
		
		# Plots...
		arrow = go.layout.Annotation(
			dict(
				x = IPD_data.loc['nanmean','Right pixel distance (m)'],
				y = threshold_percent/100,
				ax = IPD_data.loc['nanmean','Left pixel distance (m)'],
				ay = threshold_percent/100,
				xref = "x", 
				yref = "y",
				textangle = 0,
				showarrow = True,
				axref = "x", 
				ayref='y',
				arrowhead = 3,
				arrowwidth = 1.5,
			)
		)
		text = go.layout.Annotation(
			dict(
				ax = (IPD_data.loc['nanmean', 'Left pixel distance (m)']+IPD_data.loc['nanmean', 'Right pixel distance (m)'])/2,
				ay = threshold_percent/100,
				x = (IPD_data.loc['nanmean', 'Left pixel distance (m)']+IPD_data.loc['nanmean', 'Right pixel distance (m)'])/2,
				y = threshold_percent/100,
				xref = "x", 
				yref = "y",
				text = f"Inter-pixel distance: {IPD_data.loc['nanmean','Inter-pixel distance (m)']*1e6:.2f}±{IPD_data.loc['nanstd','Inter-pixel distance (m)']*1e6:.2f} µm<br> ",
				axref = "x", 
				ayref='y',
			)
		)
		
		fig = px.scatter(
			parsed_from_waveforms.join(distance, on='n_position').join(channel_positions, on='n_channel').sort_values(['n_position','channel_position']),
			x = 'Distance (m)',
			y = 'Amplitude (V) normalized',
			color = 'channel_position',
			title = f'Inter-pixel distance measurement<br><sup>Run: {bureaucrat.run_name}</sup>',
		)
		fig.update_layout(annotations=[arrow,text])
		fig.write_image( # Save it as PDF because it is too heavy as HTML.
			employee.path_to_directory_of_my_task/'IPD measurement.pdf',
		)
		
		averaged_in_position = parsed_from_waveforms[['Amplitude (V) normalized']].groupby(['n_position','n_channel','n_pulse']).agg([numpy.nanmedian, utils.kMAD])
		averaged_in_position.columns = [f'{col[0]} {col[1]}' for col in averaged_in_position.columns]
		averaged_in_position = averaged_in_position.join(distance, on='n_position')
		averaged_in_position = averaged_in_position.join(channel_positions, on='n_channel')
		
		fig = graficas_px_utils.line(
			data_frame = averaged_in_position.sort_values(['n_position','channel_position']),
			x = 'Distance (m)',
			y = 'Amplitude (V) normalized nanmedian',
			error_y = 'Amplitude (V) normalized kMAD',
			error_y_mode = 'band',
			color = 'channel_position',
			title = f'Inter-pixel distance measurement<br><sup>Run: {bureaucrat.run_name}</sup>',
		)
		fig.update_layout(annotations=[arrow,text])
		fig.write_html(
			employee.path_to_directory_of_my_task/'IPD measurement.html',
			include_plotlyjs = 'cdn',
		)
		
if __name__ == '__main__':
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
	inter_pixel_distance(
		bureaucrat = Enrique,
		number_of_bootstrapped_replicas = 11,
		force = args.force,
	)
