from the_bureaucrat.bureaucrats import RunBureaucrat # https://github.com/SengerM/the_bureaucrat
from pathlib import Path
import numpy
import pandas
from scipy.optimize import curve_fit
import warnings
import utils
from lmfit import Model
import lmfit
from scipy import special as scipy_special
import grafica.plotly_utils.utils as graficas_px_utils
import plotly.graph_objects as go

graficas_px_utils.set_my_template_as_default()

def metal_silicon_transition_model_function_left_pad(x, y_scale, laser_sigma, x_offset, y_offset, nan_policy='omit'):
	if nan_policy=='omit':
		x = x[~numpy.isnan(x)]
	else:
		raise NotImplementedError(f'`nan_policy` {repr(nan_policy)} is not implemented.')
	return y_scale*scipy_special.erf((x-x_offset)/laser_sigma*2**.5) + y_offset

def metal_silicon_transition_model_function_right_pad(x, y_scale, laser_sigma, x_offset, y_offset, nan_policy='omit'):
	return metal_silicon_transition_model_function_left_pad(-x, y_scale, laser_sigma, -x_offset, y_offset, nan_policy=nan_policy)

def fit_erf_to_TCT_1D_scan(data_df:pandas.DataFrame, window_size)->lmfit.model.ModelResult:
	"""Fit the ERF function to the left and right pad, and returns the
	results of such fit."""
	fit_results = {}
	fit_model_left_pad = Model(metal_silicon_transition_model_function_left_pad)
	fit_model_right_pad = Model(metal_silicon_transition_model_function_right_pad)
	for channel_position in set(data_df['channel_position']):
		this_pad_df = data_df.query(f'channel_position=={repr(channel_position)}')
		this_pad_df = this_pad_df.dropna()
		if channel_position == 'left':
			x_data_for_fit = this_pad_df.loc[this_pad_df['Distance (m)']<this_pad_df['Distance (m)'].mean()-window_size/2, 'Distance (m)']
			y_data_for_fit = this_pad_df.loc[this_pad_df['Distance (m)']<this_pad_df['Distance (m)'].mean()-window_size/2, 'Amplitude (V)']
			fit_model = fit_model_left_pad
		elif channel_position == 'right':
			x_data_for_fit = this_pad_df.loc[this_pad_df['Distance (m)']>this_pad_df['Distance (m)'].mean()+window_size/2, 'Distance (m)']
			y_data_for_fit = this_pad_df.loc[this_pad_df['Distance (m)']>this_pad_df['Distance (m)'].mean()+window_size/2, 'Amplitude (V)']
			fit_model = fit_model_right_pad
		parameters = fit_model.make_params(
			laser_sigma = 10e-6,
			x_offset = this_pad_df['Distance (m)'].mean()-window_size if channel_position=='left' else this_pad_df['Distance (m)'].mean()+window_size, # Transition metal→silicon in the left pad.
			y_scale = max(y_data_for_fit)/2,
			y_offset = max(y_data_for_fit)/2,
		)
		parameters['laser_sigma'].set(min=5e-6, max=22e-6)
		fit_results[channel_position] = fit_model.fit(y_data_for_fit, parameters, x=x_data_for_fit)
	return fit_results

def distance_calibration_TCT_1D_scan(bureaucrat:RunBureaucrat, window_size_meters:float, number_of_bootstrapped_replicas:int=0, force:bool=False):
	"""Perform the distance calibration using the size of the metal window
	as a reference and fitting ERF functions to each metal→silicon and 
	silicon→metal edge.
	
	Arguments
	---------
	window_size_meters: float
		Size of the opening in the metallization of the pixels through which
		the laser was shined. This is the size of the opening including the
		two pixels, i.e. the distance from metal→silicon in the left
		pixel to the silicon→metal in the right pixel. This will be used
		as reference to produce the distance calibration factor.
	"""
	Néstor = bureaucrat
	Néstor.check_these_tasks_were_run_successfully(['TCT_1D_scan','parse_waveforms','tag_channels_left_right'])
	
	if force==False and Néstor.was_task_run_successfully('distance_calibration_TCT_1D_scan'):
		return
	
	def extract_parameters_from_fit_results(fit_results:dict)->pandas.Series:
		"""Creates a series with the results of the fit, example:
		```
		channel_position  parameter_name
		left              y_scale           0.171352
						  laser_sigma       0.000011
						  x_offset          0.000031
						  y_offset          0.185701
		right             y_scale           0.161313
						  laser_sigma       0.000010
						  x_offset          0.000296
						  y_offset          0.177713
		Name: result_from_fit, dtype: float64

		```
		"""
		concat_this_one = []
		for channel_position in fit_results:
			for param in fit_results[channel_position].params:
				concat_this_one.append(
					pandas.DataFrame(
						{
							'channel_position': channel_position,
							'parameter_name': param,
							'result_from_fit': fit_results[channel_position].params[param].value,
						},
						index = [0],
					)
				)
		series = pandas.concat(concat_this_one).set_index(['channel_position','parameter_name'])['result_from_fit']
		return series
	
	with Néstor.handle_task('distance_calibration_TCT_1D_scan') as Cristina:
		parsed_from_waveforms_df = utils.load_parsed_from_waveforms_and_measured_data_in_TCT_1D_scan(Néstor)
		data_df = parsed_from_waveforms_df[['Amplitude (V)']]
		data_df = data_df.query('n_pulse==1') # Work only with the "main pulse" which has better SNR.
		
		channel_positions = pandas.read_pickle(Néstor.path_to_directory_of_task('tag_channels_left_right')/'channels_position.pickle')
		distance = utils.generate_distance_vs_n_position(parsed_from_waveforms_df)
		
		parameters_from_fits = []
		for n_bootstrap in range(number_of_bootstrapped_replicas+1):
			if n_bootstrap == 0:
				df = data_df
			else:
				df = utils.resample_by_events(data_df)
			df = df.join(channel_positions, on='n_channel')
			df = df.join(distance, on='n_position')
			fit_results = fit_erf_to_TCT_1D_scan(df, window_size=window_size_meters/2)
			_ = extract_parameters_from_fit_results(fit_results)
			_ = _.to_frame()
			_['n_bootstrap'] = n_bootstrap
			_.set_index('n_bootstrap',append=True,inplace=True)
			parameters_from_fits.append(_)
			if n_bootstrap == 0:
				fit_results_on_data = fit_results
		parameters_from_fits = pandas.concat(parameters_from_fits)
		
		results = parameters_from_fits.groupby(['channel_position','parameter_name']).agg([numpy.mean, numpy.std])
		for i in results.index:
			results.loc[i,('result_on_real_data','value')] = parameters_from_fits.loc[tuple(list(i)+[0]),'result_from_fit']
		results.to_pickle(Cristina.path_to_directory_of_my_task/'erf_fit_results.pickle')
		print(results)
		
		averaged_in_position_df = data_df[['Amplitude (V)']].groupby(['n_position','n_channel','n_pulse']).agg([numpy.nanmedian, utils.kMAD])
		averaged_in_position_df.columns = [f'{col[0]} {col[1]}' for col in averaged_in_position_df.columns]
		averaged_in_position_df = averaged_in_position_df.join(distance, on='n_position')
		averaged_in_position_df = averaged_in_position_df.join(channel_positions, on='n_channel')
		
		fig = graficas_px_utils.line(
			data_frame = averaged_in_position_df,
			x = 'Distance (m)',
			y = 'Amplitude (V) nanmedian',
			error_y = 'Amplitude (V) kMAD',
			error_y_mode = 'bands',
			color = 'channel_position',
			title = f'ERF fit for distance calibration<br><sup>Run: {Néstor.run_name}</sup>',
		)
		for channel_position in set(averaged_in_position_df['channel_position']):
			if channel_position == 'left':
				df = averaged_in_position_df.loc[averaged_in_position_df['Distance (m)']<averaged_in_position_df['Distance (m)'].mean()-window_size_meters/4,'Distance (m)']
			else:
				df = averaged_in_position_df.loc[averaged_in_position_df['Distance (m)']>averaged_in_position_df['Distance (m)'].mean()+window_size_meters/4,'Distance (m)']
			x = numpy.linspace(min(df), max(df), 99)
			fig.add_trace(
				go.Scatter(
					x = x,
					y = fit_results_on_data[channel_position].eval(params=fit_results_on_data[channel_position].params, x = x),
					mode = 'lines',
					name = f'Fit erf {channel_position} pad, σ<sub>laser</sub>={results.loc[(channel_position,"laser_sigma"),("result_on_real_data","value")]*1e6:.2f} ± {results.loc[(channel_position,"laser_sigma"),("result_from_fit","std")]*1e6:.2f} µm',
					line = dict(color='black', dash='dash'),
				)
			)
		fig.write_html(
			Cristina.path_to_directory_of_my_task/'ERF fit.html',
			include_plotlyjs = 'cdn',
		)
		
		raise NotImplementedError()

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
	distance_calibration_TCT_1D_scan(
		bureaucrat = Enrique,
		force = args.force,
		window_size_meters = 250e-6,
		number_of_bootstrapped_replicas = 11,
	)

