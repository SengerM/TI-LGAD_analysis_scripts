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
from uncertainties import ufloat

def metal_silicon_transition_model_function_left_pad(x, y_scale, laser_sigma, x_offset, y_offset, nan_policy='omit'):
	if nan_policy=='omit':
		x = x[~numpy.isnan(x)]
	else:
		raise NotImplementedError(f'`nan_policy` {repr(nan_policy)} is not implemented.')
	return y_scale*scipy_special.erf((x-x_offset)/laser_sigma*2**.5) + y_offset

def metal_silicon_transition_model_function_right_pad(x, y_scale, laser_sigma, x_offset, y_offset, nan_policy='omit'):
	return metal_silicon_transition_model_function_left_pad(-x, y_scale, laser_sigma, -x_offset, y_offset, nan_policy=nan_policy)

def calibrate_distance(distance:pandas.Series, calibration_factors:pandas.Series)->pandas.Series:
	"""Given a series with the distance and the calibration factors produced
	by `distance_calibration_TCT_1D_scan`, this functions calculates
	and returns the calibrated distance.
	
	Example
	-------
	distance:
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
	calibration_factors:
	```
	multiply_distance_by_this_to_calibrate                 0.942228
	subtract_this_to_distance_to_center_measured_window    0.000163
	dtype: float64
	```
	"""
	if distance.name != 'Distance (m)':
		raise ValueError(f'I was expecting a series called "Distance (m)" in `distance`...')
	d = (distance-calibration_factors['subtract_this_to_distance_to_center_measured_window'])*calibration_factors['multiply_distance_by_this_to_calibrate']
	d.rename('Calibrated distance (m)', inplace=True)
	return d

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
	Néstor.check_these_tasks_were_run_successfully(['TCT_1D_scan','parse_waveforms','tag_channels_left_right','normalization_factors_for_amplitude'])
	
	if force==False and Néstor.was_task_run_successfully('distance_calibration_TCT_1D_scan'):
		return
	
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
				y_data_for_fit = this_pad_df.loc[this_pad_df['Distance (m)']<this_pad_df['Distance (m)'].mean()-window_size/2, 'Amplitude (V) normalized']
				fit_model = fit_model_left_pad
			elif channel_position == 'right':
				x_data_for_fit = this_pad_df.loc[this_pad_df['Distance (m)']>this_pad_df['Distance (m)'].mean()+window_size/2, 'Distance (m)']
				y_data_for_fit = this_pad_df.loc[this_pad_df['Distance (m)']>this_pad_df['Distance (m)'].mean()+window_size/2, 'Amplitude (V) normalized']
				fit_model = fit_model_right_pad
			parameters = fit_model.make_params(
				laser_sigma = 10e-6,
				x_offset = this_pad_df['Distance (m)'].mean()-window_size if channel_position=='left' else this_pad_df['Distance (m)'].mean()+window_size, # Transition metal→silicon in the left pad.
				y_scale = 1/2,
				y_offset = 1/2,
			)
			parameters['y_scale'].set(min=.1, max=.9)
			parameters['y_offset'].set(min=.1, max=.9)
			parameters['laser_sigma'].set(min=5e-6, max=22e-6)
			fit_results[channel_position] = fit_model.fit(y_data_for_fit, parameters, x=x_data_for_fit)
		return fit_results
	
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
		
		data_df = utils.normalize_data(
			parsed_from_waveforms_df[['Amplitude (V)']], 
			pandas.read_pickle(Cristina.path_to_directory_of_task('normalization_factors_for_amplitude')/'normalization_data.pickle')
		)
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
		
		x_offset = {}
		for channel_position in {'left','right'}:
			x_offset[channel_position] = ufloat(results.loc[(channel_position,'x_offset'),('result_from_fit','mean')], results.loc[(channel_position,'x_offset'),('result_from_fit','std')])
		measured_window_size = x_offset['right']-x_offset['left']
		measured_window_center = (x_offset['right']+x_offset['left'])/2
		
		multiply_distance_by_this_to_calibrate = window_size_meters/measured_window_size
		s = pandas.Series(
			{
				'multiply_distance_by_this_to_calibrate': (window_size_meters/measured_window_size).nominal_value,
				'subtract_this_to_distance_to_center_measured_window': measured_window_center.nominal_value
			}
		)
		s.to_pickle(Cristina.path_to_directory_of_my_task/'distance_calibration_factors.pickle')
		
		averaged_in_position_df = data_df[['Amplitude (V) normalized']].groupby(['n_position','n_channel','n_pulse']).agg([utils.FUNCTION_TO_USE_FOR_AVERAGE, utils.FUNCTION_TO_USE_FOR_FLUCTUATIONS])
		averaged_in_position_df.columns = [' '.join(col) for col in averaged_in_position_df.columns]
		averaged_in_position_df.rename(
			columns = {
				f'Amplitude (V) normalized {utils.FUNCTION_TO_USE_FOR_AVERAGE.__name__}': 'Normalized amplitude',
				f'Amplitude (V) normalized {utils.FUNCTION_TO_USE_FOR_FLUCTUATIONS.__name__}': 'Normalized amplitude error',
			},
			inplace = True,
		)
		averaged_in_position_df = averaged_in_position_df.join(distance, on='n_position')
		averaged_in_position_df = averaged_in_position_df.join(channel_positions, on='n_channel')
		
		averaged_in_position_df = averaged_in_position_df.join(calibrate_distance(distance, s), on='n_position')
		
		fig = graficas_px_utils.line(
			data_frame = averaged_in_position_df.reset_index(drop=False).sort_values(['channel_position','n_position']),
			x = 'Calibrated distance (m)',
			y = 'Normalized amplitude',
			error_y = 'Normalized amplitude error',
			error_y_mode = 'bands',
			color = 'channel_position',
			title = f'Normalized amplitude after distance calibration<br><sup>Run: {Néstor.run_name}</sup>',
		)
		fig.write_html(
			Cristina.path_to_directory_of_my_task/'plot with calibrated distance.html',
			include_plotlyjs = 'cdn',
		)
		
		fig = graficas_px_utils.line(
			data_frame = averaged_in_position_df.reset_index(drop=False).sort_values(['channel_position','n_position']),
			x = 'Distance (m)',
			y = 'Normalized amplitude',
			error_y = 'Normalized amplitude error',
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
					name = f'Fit erf {channel_position} pad, σ<sub>laser</sub>={results.loc[(channel_position,"laser_sigma"),("result_on_real_data","value")]*1e6:.2f}±{results.loc[(channel_position,"laser_sigma"),("result_from_fit","std")]*1e6:.2f} µm',
					line = dict(color='black', dash='dash'),
				)
			)
		fig.add_vline(
			x = measured_window_center.nominal_value,
			annotation_text = f'Center according to ERF fits {measured_window_center*1e6:.2f} µm'.replace('+/-','±'),
			annotation_textangle = -90,
			line = dict(width = .5),
		)
		for channel_position in {'left','right'}:
			sign = 1 if channel_position=='right' else -1
			fig.add_vline(
				x = (measured_window_center+sign*measured_window_size/2).nominal_value,
				line = dict(width = .5),
			)
		arrow = go.layout.Annotation(
			dict(
				x = (measured_window_center+measured_window_size/2).nominal_value,
				y = .5,
				ax = (measured_window_center-measured_window_size/2).nominal_value,
				ay = .5,
				xref = "x", 
				yref = "y",
				text = f'Measured window size: {measured_window_size*1e6:.2f}µm'.replace('+/-','±'),
				textangle = 0,
				showarrow = True,
				axref = "x", 
				ayref='y',
				arrowhead = 3,
				arrowwidth = 1.5,
			)
		)
		fig.update_layout(annotations=[arrow])
		fig.write_html(
			Cristina.path_to_directory_of_my_task/'ERF fit.html',
			include_plotlyjs = 'cdn',
		)
		
if __name__=='__main__':
	import argparse
	
	graficas_px_utils.set_my_template_as_default()
	
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

