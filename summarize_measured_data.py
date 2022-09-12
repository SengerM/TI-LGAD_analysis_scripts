import utils
from the_bureaucrat.bureaucrats import RunBureaucrat # https://github.com/SengerM/the_bureaucrat
import pandas
from huge_dataframe.SQLiteDataFrame import load_whole_dataframe # https://github.com/SengerM/huge_dataframe
from pathlib import Path
import numpy
import warnings
import multiprocessing
import plotly.express as px
from grafica.plotly_utils.utils import set_my_template_as_default

set_my_template_as_default()

def summarize_measured_data(bureaucrat:RunBureaucrat, force:bool=False):
	"""Produces a lightweight summary of the "measured data" in the `TCT_1D_scan` task
	that does not belong to the waveforms, such as the temperature, bias voltage,
	bias current, etc.
	"""
	Paul = bureaucrat
	Paul.check_these_tasks_were_run_successfully('TCT_1D_scan')
	
	if Paul.was_task_run_successfully('summarize_measured_data') and force==False:
		return
	
	with Paul.handle_task('summarize_measured_data') as Pauls_employee:
		measured_data_df = load_whole_dataframe(Pauls_employee.path_to_directory_of_task('TCT_1D_scan')/'measured_data.sqlite')
		summary_df = measured_data_df.drop(columns='When').agg([numpy.nanmean, numpy.nanstd, utils.kMAD, numpy.nanmedian, numpy.nanmax, numpy.nanmin])
		summary_df.to_pickle(Pauls_employee.path_to_directory_of_my_task/'summary.pickle')

def read_summarized_data_in_TCT_1D_scan_sweeping_bias_voltage(bureaucrat:RunBureaucrat):
	"""Read all the summarized data from each subrun and puts it together
	in a single dataframe. 
	
	Example
	-------
	```
														 x (m)                                            ... Humidity (%RH)                                    
													   nanmean nanstd kMAD nanmedian    nanmax    nanmin  ...        nanmean nanstd kMAD nanmedian nanmax nanmin
	run_name                                                                                              ...                                                   
	20220908134101_hate_the_bugged_oscilloscope_190V -0.002926    0.0  0.0 -0.002926 -0.002926 -0.002926  ...            0.0    0.0  0.0       0.0    0.0    0.0
	20220908134101_hate_the_bugged_oscilloscope_220V -0.002926    0.0  0.0 -0.002926 -0.002926 -0.002926  ...            0.0    0.0  0.0       0.0    0.0    0.0
	20220908134101_hate_the_bugged_oscilloscope_205V -0.002926    0.0  0.0 -0.002926 -0.002926 -0.002926  ...            0.0    0.0  0.0       0.0    0.0    0.0
	20220908134101_hate_the_bugged_oscilloscope_160V -0.002926    0.0  0.0 -0.002926 -0.002926 -0.002926  ...            0.0    0.0  0.0       0.0    0.0    0.0
	20220908134101_hate_the_bugged_oscilloscope_197V -0.002926    0.0  0.0 -0.002926 -0.002926 -0.002926  ...            0.0    0.0  0.0       0.0    0.0    0.0
	20220908134101_hate_the_bugged_oscilloscope_175V -0.002926    0.0  0.0 -0.002926 -0.002926 -0.002926  ...            0.0    0.0  0.0       0.0    0.0    0.0
	20220908134101_hate_the_bugged_oscilloscope_212V -0.002926    0.0  0.0 -0.002926 -0.002926 -0.002926  ...            0.0    0.0  0.0       0.0    0.0    0.0
	20220908134101_hate_the_bugged_oscilloscope_100V -0.002926    0.0  0.0 -0.002926 -0.002926 -0.002926  ...            0.0    0.0  0.0       0.0    0.0    0.0
	20220908134101_hate_the_bugged_oscilloscope_145V -0.002926    0.0  0.0 -0.002926 -0.002926 -0.002926  ...            0.0    0.0  0.0       0.0    0.0    0.0
	20220908134101_hate_the_bugged_oscilloscope_130V -0.002926    0.0  0.0 -0.002926 -0.002926 -0.002926  ...            0.0    0.0  0.0       0.0    0.0    0.0
	20220908134101_hate_the_bugged_oscilloscope_115V -0.002926    0.0  0.0 -0.002926 -0.002926 -0.002926  ...            0.0    0.0  0.0       0.0    0.0    0.0
	20220908134101_hate_the_bugged_oscilloscope_182V -0.002926    0.0  0.0 -0.002926 -0.002926 -0.002926  ...            0.0    0.0  0.0       0.0    0.0    0.0

	[12 rows x 54 columns]
	```
	"""
	Palpatine = bureaucrat
	list_of_summaries = []
	for subrun_bureaucrat in Palpatine.list_subruns_of_task('TCT_1D_scan_sweeping_bias_voltage'):
		subrun_bureaucrat.check_these_tasks_were_run_successfully('summarize_measured_data')
		df = pandas.read_pickle(subrun_bureaucrat.path_to_directory_of_task('summarize_measured_data')/'summary.pickle')
		df = df.unstack().to_frame().transpose()
		df['run_name'] = subrun_bureaucrat.run_name
		df.set_index('run_name', inplace=True)
		list_of_summaries.append(df)
	summary_df = pandas.concat(list_of_summaries)
	return summary_df
