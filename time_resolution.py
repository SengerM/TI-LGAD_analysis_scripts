from the_bureaucrat.bureaucrats import RunBureaucrat # https://github.com/SengerM/the_bureaucrat
from pathlib import Path
from huge_dataframe.SQLiteDataFrame import SQLiteDataFrameDumper, load_whole_dataframe # https://github.com/SengerM/huge_dataframe
from utils import integrate_distance_given_path, kMAD, generate_distance_vs_n_position, load_parsed_from_waveforms_and_measured_data_in_TCT_1D_scan, resample_by_events
from grafica.plotly_utils.utils import line, set_my_template_as_default # https://github.com/SengerM/grafica
import numpy
import pandas
from scipy.optimize import curve_fit
import warnings
import utils
import plotly.express as px
import summarize_measured_data

def gaussian(x, mu, sigma, amplitude=1):
	return amplitude/sigma/(2*numpy.pi)**.5*numpy.exp(-((x-mu)/sigma)**2/2)

def plot_statistics_vs_distance(df, column_name):
	averaged_in_position_df = df.groupby(['n_position','n_channel','n_pulse']).agg([utils.FUNCTION_TO_USE_FOR_AVERAGE, utils.FUNCTION_TO_USE_FOR_FLUCTUATIONS])
	averaged_in_position_df.columns = [' '.join(col) for col in averaged_in_position_df.columns]
	averaged_in_position_df.rename(
		columns = {col: col.replace(f' {utils.FUNCTION_TO_USE_FOR_AVERAGE.__name__}','').replace(f' {utils.FUNCTION_TO_USE_FOR_FLUCTUATIONS.__name__}') + (' error' if utils.FUNCTION_TO_USE_FOR_FLUCTUATIONS.__name__ in col else '') for col in averaged_in_position_df.columns},
		inplace = True,
	)
	return line(
		data_frame = averaged_in_position_df.reset_index(drop=False),
		x = 'Distance (m)',
		y = column_name,
		error_y = f'{column_name} error',
		error_y_mode = 'bands',
		color = 'n_channel',
		line_dash = 'n_pulse',
		labels = utils.LABELS_FOR_PLOTS,
	)

def calculate_Δt(data_df:pandas.DataFrame)->pandas.DataFrame:
	"""Given a data frame with the columns containing the time at which
	the signal crossed the different CFD thresholds, i.e. the columns
	`'t_10 (s)', 't_20 (s)', 't_30 (s)', 't_40 (s)', 't_50 (s)', 't_60 (s)', 't_70 (s)', 't_80 (s)', 't_90 (s)'`
	and also one index column `'n_pulse'`, this function calculates the 
	Δt for all pairs of `f't_{whatever} (s)'`.
	
	Example input:
	```
												t_10 (s)      t_20 (s)      t_30 (s)  ...      t_80 (s)      t_90 (s)
	n_position n_trigger n_channel n_pulse                                            ...                            
	0          0         1         1        1.441200e-07  1.441344e-07  1.441489e-07  ...  1.442211e-07  1.442356e-07
								   2        2.579618e-07  2.579772e-07  2.579925e-07  ...  2.586193e-07  2.586346e-07
						 4         1        2.158016e-07  2.158125e-07  2.158235e-07  ...  2.160563e-07  2.160781e-07
								   2        3.037996e-07  3.039582e-07  3.039665e-07  ...  3.040166e-07  3.040333e-07
			   1         1         1        1.601666e-07  1.601815e-07  1.601963e-07  ...  1.605852e-07  1.605926e-07
	...                                              ...           ...           ...  ...           ...           ...
	333        109       4         2        2.407565e-07  2.407669e-07  2.407773e-07  ...  2.409792e-07  2.409896e-07
			   110       1         1        2.119685e-07  2.119886e-07  2.121088e-07  ...  2.123097e-07  2.123298e-07
								   2        2.600135e-07  2.600342e-07  2.602525e-07  ...  2.603086e-07  2.603293e-07
						 4         1        2.154556e-07  2.154661e-07  2.154766e-07  ...  2.155290e-07  2.155395e-07
								   2        2.602517e-07  2.602682e-07  2.602847e-07  ...  2.603671e-07  2.603835e-07
	```
	and example output:
	```
														  Δt (s)
	n_position n_trigger n_channel k1_percent k2_percent              
	0          0         1         10         10         -1.138418e-07
						 4         10         10         -8.799804e-08
			   1         1         10         10         -9.049728e-08
						 4         10         10         -1.742459e-07
			   2         1         10         10         -1.312444e-07
	...                                                            ...
	333        108       4         90         90         -2.960782e-08
			   109       1         90         90         -6.920527e-08
						 4         90         90         -9.640726e-08
			   110       1         90         90         -4.799945e-08
						 4         90         90         -4.484402e-08
	```
	"""
	if 'n_pulse' not in data_df.index.names:
		raise ValueError(f'I was expecting `n_pulse` to be in the index of `data_df` but it is not there.')
	REQUIRED_COLUMNS = {'t_10 (s)', 't_20 (s)', 't_30 (s)', 't_40 (s)', 't_50 (s)', 't_60 (s)', 't_70 (s)', 't_80 (s)', 't_90 (s)'}
	if len(REQUIRED_COLUMNS-set(data_df.columns)) != 0:
		raise ValueError(f'`data_df` must possess the columns {sorted(REQUIRED_COLUMNS)} but {sorted(REQUIRED_COLUMNS-set(data_df.columns))} cannot be found within its columns.')
	delta_tes_list = []
	pulse_1 = data_df.query('n_pulse==1').droplevel('n_pulse',axis=0)
	pulse_2 = data_df.query('n_pulse==2').droplevel('n_pulse',axis=0)
	for k1 in [10,20,30,40,50,60,70,80,90]:
		for k2 in [10,20,30,40,50,60,70,80,90]:
			df = pandas.DataFrame(index=pulse_1.index)
			df['Δt (s)'] = pulse_1[f't_{k1} (s)'] - pulse_2[f't_{k2} (s)']
			df['k1_percent'] = k1
			df['k2_percent'] = k2
			df.set_index(['k1_percent','k2_percent'], append=True, inplace=True)
			delta_tes_list.append(df)
	return pandas.concat(delta_tes_list)

def fit_gaussian_to_samples(samples, bins='auto', nan_policy='drop'):
	if nan_policy == 'drop':
		samples = samples[~numpy.isnan(samples)]
	else:
		raise NotImplementedError(f'`nan_policy={nan_policy}` not implemented.')
	hist, bins_edges = numpy.histogram(
		samples,
		bins = bins,
	)
	x_values = bins_edges[:-1] + numpy.diff(bins_edges)[0]/2
	y_values = hist
	try:
		popt, pcov = curve_fit(
			gaussian,
			x_values,
			y_values,
			p0 = [numpy.median(samples),kMAD(samples),max(y_values)],
		)
		return popt[0], popt[1], popt[2]
	except RuntimeError: # This happens when the fit fails because there are very few samples.
		return float('NaN'),float('NaN'),float('NaN')

def sigma_from_gaussian_fit(x, nan_policy='drop')->float:
	"""Estimates the standard deviation of samples `x` by fitting a 
	Gaussian function and getting the value of sigma."""
	with warnings.catch_warnings():
		warnings.filterwarnings("ignore", message="divide by zero encountered in divide")
		_, sigma, _ = fit_gaussian_to_samples(samples=x, bins='auto', nan_policy=nan_policy)
	return sigma

def jitter_vs_distance_in_TCT_1D_scan(bureaucrat:RunBureaucrat, number_of_bootstrapped_replicas:int=0, force:bool=False):
	Nicanor = bureaucrat
	
	Nicanor.check_these_tasks_were_run_successfully(['TCT_1D_scan','parse_waveforms'])
	
	if force==False and Nicanor.was_task_run_successfully('jitter_vs_distance_in_TCT_1D_scan'):
		return
	
	with Nicanor.handle_task('jitter_vs_distance_in_TCT_1D_scan') as Nicanors_employee:
		data_df = load_parsed_from_waveforms_and_measured_data_in_TCT_1D_scan(Nicanor)
		data_df['Distance (m)'] = integrate_distance_given_path(list(data_df[['x (m)', 'y (m)', 'z (m)']].to_numpy()))
		
		data_df = data_df[['Distance (m)', 't_10 (s)', 't_20 (s)', 't_30 (s)', 't_40 (s)', 't_50 (s)', 't_60 (s)', 't_70 (s)', 't_80 (s)', 't_90 (s)']]
		
		jitters_list = []
		for n_bootstrap in range(number_of_bootstrapped_replicas+1):
			if n_bootstrap == 0:
				df = data_df
			else:
				df = resample_by_events(data_df)
			Δt_df = calculate_Δt(df)
			jitter = Δt_df.groupby(level=['n_position','n_channel','k1_percent','k2_percent']).agg(func=kMAD) # Here the best would be to use the Gaussian fit, but it takes too long. After a few iterations I noticed that kMAD produces much better results than std, due to outliers.
			jitter.rename(columns={'Δt (s)': 'Jitter (s)'}, inplace=True)
			if n_bootstrap == 0:
				jitter_df = jitter.copy()
			jitter['n_bootstrap'] = n_bootstrap
			jitter.set_index('n_bootstrap', append=True, inplace=True)
			jitters_list.append(jitter)
		jitters_bootstrapped_df = pandas.concat(jitters_list)
		
		jitter_df = jitters_bootstrapped_df.groupby(['n_position','n_channel','k1_percent','k2_percent']).agg([numpy.nanmean, numpy.nanstd])
		jitter_df.columns = [' '.join(col) for col in jitter_df.columns]
		jitter_df = jitter_df.rename(columns = {'Jitter (s) nanmean': 'Jitter (s)', 'Jitter (s) nanstd': 'Jitter (s) error'})
		
		with SQLiteDataFrameDumper(Nicanors_employee.path_to_directory_of_my_task/Path('jitter.sqlite'), dump_after_n_appends=100) as dumper:
			dumper.append(jitter_df)
		
		distance = data_df.droplevel(list(set(data_df.index.names)-{'n_position'}),axis=0)['Distance (m)']
		distance = distance[~distance.index.duplicated(keep='first')]
		distance = distance.to_frame()
		jitter_df = jitter_df.merge(distance, left_index=True, right_index=True)
		
		fig = line(
			data_frame = jitter_df.query('`k1_percent`==20').query('`k2_percent`==20').query('`Jitter (s)`<200e-12').reset_index(drop=False).sort_values('n_position'),
			x = 'Distance (m)',
			y = 'Jitter (s)',
			error_y = 'Jitter (s) error',
			error_y_mode = 'bands',
			color = 'n_channel',
			title = f'Jitter vs position<br><sup>Run: {Nicanor.run_name}</sup>',
			labels = utils.LABELS_FOR_PLOTS,
		)
		fig.write_html(
			str(Nicanors_employee.path_to_directory_of_my_task/'jitter_vs_distance.html'),
			include_plotlyjs = 'cdn',
		)

def time_resolution_vs_distance_in_TCT_1D_scan(bureaucrat:RunBureaucrat, cfd_thresholds:tuple, force:bool=False):
	Rick = bureaucrat
	
	Rick.check_these_tasks_were_run_successfully(['jitter_vs_distance_in_TCT_1D_scan', 'TCT_1D_scan'])
	
	if force==False and Rick.was_task_run_successfully('time_resolution_vs_distance_in_TCT_1D_scan'):
		return
	
	with Rick.handle_task('time_resolution_vs_distance_in_TCT_1D_scan') as Ricks_employee:
		jitter_df = load_whole_dataframe(Rick.path_to_directory_of_task('jitter_vs_distance_in_TCT_1D_scan')/'jitter.sqlite')
		data_df = load_parsed_from_waveforms_and_measured_data_in_TCT_1D_scan(Rick)
		jitter_df = jitter_df.join(generate_distance_vs_n_position(data_df[['x (m)', 'y (m)', 'z (m)']]), on='n_position')
		
		for col in jitter_df.columns:
			if 'jitter' in col.lower():
				jitter_df[col] /= 2**.5
				jitter_df.rename(columns={col: col.replace('Jitter','Time resolution')}, inplace=True)
		
		time_resolution = jitter_df.loc[:,:,cfd_thresholds[0],cfd_thresholds[1]]
		
		time_resolution.to_pickle(Ricks_employee.path_to_directory_of_my_task/'time_resolution.pickle')
		
		fig = line(
			data_frame = time_resolution.query('`Time resolution (s)`<200e-12').reset_index(drop=False).sort_values(['n_position','n_channel']),
			x = 'Distance (m)',
			y = 'Time resolution (s)',
			error_y = 'Time resolution (s) error',
			error_y_mode = 'bands',
			color = 'n_channel',
			title = f'Time resolution vs position<br><sup>Run: {Rick.run_name}</sup>',
			labels = utils.LABELS_FOR_PLOTS,
		)
		fig.write_html(
			str(Ricks_employee.path_to_directory_of_my_task/'time_resolution_vs_distance.html'),
			include_plotlyjs = 'cdn',
		)

def pixel_time_resolution(bureaucrat:RunBureaucrat, approximate_window_size_meters:float, approximate_laser_size_meters:float, force:bool=False):
	"""Calculates a time resolution as a single number representative of
	DUT assuming the time resolution is uniform along the active area."""
	bureaucrat.check_these_tasks_were_run_successfully(['time_resolution_vs_distance_in_TCT_1D_scan','tag_channels_left_right'])
	
	if force==False and bureaucrat.was_task_run_successfully('pixel_time_resolution'):
		return
	
	with bureaucrat.handle_task('pixel_time_resolution') as employee:
		time_resolution_data = pandas.read_pickle(employee.path_to_directory_of_task('time_resolution_vs_distance_in_TCT_1D_scan')/'time_resolution.pickle')
		
		channel_positions = pandas.read_pickle(employee.path_to_directory_of_task('tag_channels_left_right')/'channels_position.pickle')
		
		cfd_thresholds_used = {k: list(set(time_resolution_data.index.get_level_values(k)))[0] for k in {'k1_percent','k2_percent'}}
		time_resolution_data = time_resolution_data.droplevel('k1_percent')
		time_resolution_data = time_resolution_data.droplevel('k2_percent')
		
		distance = time_resolution_data['Distance (m)']
		distance = distance[~distance.index.duplicated(keep='first')]
		
		time_resolution_data = time_resolution_data.join(channel_positions, on='n_channel')
		
		distance_left_window_ends = distance.mean()-approximate_laser_size_meters
		distance_left_window_begins = distance.mean()-approximate_window_size_meters/2+approximate_laser_size_meters
		distance_right_window_begins = distance.mean()+approximate_laser_size_meters
		distance_right_window_ends = distance.mean()+approximate_window_size_meters/2-approximate_laser_size_meters
		
		_ = time_resolution_data
		places_with_data_of_interest =  (_['channel_position']=='left')&(_['Distance (m)']>distance_left_window_begins)&(_['Distance (m)']<distance_left_window_ends)
		places_with_data_of_interest |= (_['channel_position']=='right')&(_['Distance (m)']>distance_right_window_begins)&(_['Distance (m)']<distance_right_window_ends)
		time_resolution_final_result = time_resolution_data.loc[places_with_data_of_interest]
		time_resolution_final_result = time_resolution_final_result.agg([numpy.mean, numpy.std])
		
		time_resolution_final_result = pandas.Series(
			{
				'Time resolution (s)': time_resolution_final_result.loc['mean','Time resolution (s)'],
				'Time resolution (s) error': time_resolution_final_result.loc['std','Time resolution (s)'],
			}
		)
		time_resolution_final_result.to_pickle(employee.path_to_directory_of_my_task/'time_resolution.pickle')
		
		fig = line(
			data_frame = time_resolution_data.query('`Time resolution (s)`<200e-12').query('`Time resolution (s) error`<100e-12').reset_index(drop=False).sort_values(['n_position','n_channel']),
			x = 'Distance (m)',
			y = 'Time resolution (s)',
			error_y = 'Time resolution (s) error',
			error_y_mode = 'bands',
			color = 'channel_position',
			title = f'TCT time resolution<br><sup>Run: {bureaucrat.run_name}</sup>',
			labels = utils.LABELS_FOR_PLOTS,
		)
		fig.add_hrect(
			y0 = time_resolution_final_result['Time resolution (s)']-time_resolution_final_result['Time resolution (s) error'],
			y1 = time_resolution_final_result['Time resolution (s)']+time_resolution_final_result['Time resolution (s) error'],
			opacity = 0.2,
			line_width = 0, 
			fillcolor = "black",
		)
		fig.add_hline(
			y = time_resolution_final_result['Time resolution (s)'],
			annotation_text = f"Time resolution = {time_resolution_final_result['Time resolution (s)']*1e12:.2f}±{time_resolution_final_result['Time resolution (s) error']*1e12:.2f} ps",
			# ~ annotation_font_color = '#31ba2f',
		)
		fig.update_yaxes(range=[0, 100e-12])
		fig.write_html(
			str(employee.path_to_directory_of_my_task/'time_resolution.html'),
			include_plotlyjs = 'cdn',
		)

def time_resolution_vs_bias_voltage(bureaucrat:RunBureaucrat):
	"""Collects the time resolution at each voltage for a `TCT_1D_scan_sweeping_bias_voltage`"""
	bureaucrat.check_these_tasks_were_run_successfully('TCT_1D_scan_sweeping_bias_voltage')
	concat_this = []
	for subrun_bureaucrat in bureaucrat.list_subruns_of_task('TCT_1D_scan_sweeping_bias_voltage'):
		subrun_bureaucrat.check_these_tasks_were_run_successfully(['pixel_time_resolution'])
		s = pandas.read_pickle(subrun_bureaucrat.path_to_directory_of_task('pixel_time_resolution')/'time_resolution.pickle')
		s['run_name'] = subrun_bureaucrat.run_name
		df = pandas.DataFrame(s.to_dict(), index=[0]).set_index('run_name')
		concat_this.append(df)
	df = pandas.concat(concat_this)
	measured_data = summarize_measured_data.read_summarized_data_in_TCT_1D_scan_sweeping_bias_voltage(bureaucrat)
	measured_data.drop(columns=[c for c in measured_data.columns if 'Bias voltage (V)'!=c[0]], inplace=True)
	df['Bias voltage (V)'] = measured_data[('Bias voltage (V)','nanmean')]
	df['Bias voltage (V) error'] = measured_data[('Bias voltage (V)','nanstd')]
	with bureaucrat.handle_task('time_resolution_vs_bias_voltage') as employee:
		df.to_pickle(employee.path_to_directory_of_my_task/'time_resolution_vs_bias_voltage.pickle')
	
		fig = px.line(
			df.sort_values('Bias voltage (V)'),
			x = 'Bias voltage (V)',
			error_x = 'Bias voltage (V) error',
			y = 'Time resolution (s)',
			error_y = 'Time resolution (s) error',
			markers = True,
			title = f'Time resolution vs bias voltage in TCT<br><sup>Run: {bureaucrat.run_name}</sup>',
			labels = utils.LABELS_FOR_PLOTS,
		)
		fig.update_xaxes(autorange="reversed")
		fig.write_html(
			employee.path_to_directory_of_my_task/'time_resolution_vs_bias_voltage.html',
			include_plotlyjs = 'cdn',
		)
	
if __name__ == '__main__':
	import argparse
	
	set_my_template_as_default()
	
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
	pixel_time_resolution(
		bureaucrat = Enrique,
		force = args.force,
		approximate_window_size_meters = 250e-6,
		approximate_laser_size_meters = 10e-6,
	)
