from the_bureaucrat.bureaucrats import RunBureaucrat # https://github.com/SengerM/the_bureaucrat
from pathlib import Path
import utils
import distance_calibration
import summarize_measured_data
import time_resolution
import multiprocessing
import inter_pixel_distance
import grafica.plotly_utils.utils

def run_tasks_on_TCT_1D_scan(bureaucrat:RunBureaucrat, force:bool=False, silent:bool=True, skip_timing_analysis:bool=False):
	"""Runs ALL the analysis tasks for a TCT 1D scan on a TI-LGAD."""
	bureaucrat.check_these_tasks_were_run_successfully(['TCT_1D_scan','parse_waveforms'])
	
	WINDOW_SIZE_METERS = 250e-6
	LASER_SIZE_METERS = 11e-6
	
	if not silent:
		print(f'Running `summarize_measured_data.summarize_measured_data` on {bureaucrat.run_name}...')
	summarize_measured_data.summarize_measured_data(
		bureaucrat = bureaucrat,
		force = force,
	)
	if not silent:
		print(f'Running `utils.tag_channels_left_right` on {bureaucrat.run_name}...')
	utils.tag_channels_left_right(
		bureaucrat = bureaucrat,
		force = force,
	)
	if not silent:
		print(f'Running `utils.normalization_factors_for_amplitude` on {bureaucrat.run_name}...')
	utils.normalization_factors_for_amplitude(
		bureaucrat = bureaucrat,
		approximate_window_size_meters = WINDOW_SIZE_METERS, 
		approximate_laser_size_meters = LASER_SIZE_METERS, 
		force = force,
	)
	if not silent:
		print(f'Running `distance_calibration.distance_calibration_TCT_1D_scan` on {bureaucrat.run_name}...')
	distance_calibration.distance_calibration_TCT_1D_scan(
		bureaucrat = bureaucrat,
		window_size_meters = WINDOW_SIZE_METERS,
		number_of_bootstrapped_replicas = 11,
		force = force,
	)
	if skip_timing_analysis == False:
		if not silent:
			print(f'Running `time_resolution.jitter_vs_distance_in_TCT_1D_scan` on {bureaucrat.run_name}...')
		time_resolution.jitter_vs_distance_in_TCT_1D_scan(
			bureaucrat = bureaucrat,
			number_of_bootstrapped_replicas = 11,
			force = force,
		)
		if not silent:
			print(f'Running `time_resolution.time_resolution_vs_distance_in_TCT_1D_scan` on {bureaucrat.run_name}...')
		time_resolution.time_resolution_vs_distance_in_TCT_1D_scan(
			bureaucrat = bureaucrat,
			cfd_thresholds = (20,20),
			force = force,
		)
		if not silent:
			print(f'Running `time_resolution.pixel_time_resolution` on {bureaucrat.run_name}...')
		time_resolution.pixel_time_resolution(
			bureaucrat = bureaucrat,
			approximate_window_size_meters = WINDOW_SIZE_METERS,
			approximate_laser_size_meters = LASER_SIZE_METERS,
			force = force,
		)
	if not silent:
		print(f'Running `inter_pixel_distance.inter_pixel_distance` on {bureaucrat.run_name}...')
	inter_pixel_distance.inter_pixel_distance(
		bureaucrat = bureaucrat,
		number_of_bootstrapped_replicas = 11,
		threshold_percent = 50,
		force = force,
	)

def run_tasks_on_TCT_1D_scan_sweeping_bias_voltage(bureaucrat:RunBureaucrat, number_of_processes:int=1, silent:bool=True, force:bool=False, skip_timing_analysis:bool=False):
	"""Runs ALL the analysis fora TCT 1D scan on a TI-LGAD."""
	bureaucrat.check_these_tasks_were_run_successfully('TCT_1D_scan_sweeping_bias_voltage')
	subruns = bureaucrat.list_subruns_of_task('TCT_1D_scan_sweeping_bias_voltage')
	
	with multiprocessing.Pool(number_of_processes) as p:
		p.starmap(
			run_tasks_on_TCT_1D_scan,
			[(bur,frc,slnt,skp_tmng) for bur,frc,slnt,skp_tmng in zip(subruns, [force]*len(subruns), [silent]*len(subruns), [skip_timing_analysis]*len(subruns))]
		)
	if skip_timing_analysis == False:
		time_resolution.time_resolution_vs_bias_voltage(bureaucrat)
	inter_pixel_distance.inter_pixel_distance_vs_bias_voltage(bureaucrat)

def main(bureaucrat:RunBureaucrat, force:bool=False, skip_timing_analysis:bool=False, number_of_processes:int=1):
	if bureaucrat.was_task_run_successfully('TCT_1D_scan'):
		run_tasks_on_TCT_1D_scan(
			bureaucrat = bureaucrat,
			force = force,
			silent = False,
			skip_timing_analysis = skip_timing_analysis,
		)
	elif bureaucrat.was_task_run_successfully('TCT_1D_scan_sweeping_bias_voltage'):
		run_tasks_on_TCT_1D_scan_sweeping_bias_voltage(
			bureaucrat = bureaucrat,
			number_of_processes = number_of_processes,
			force = force,
			silent = False,
			skip_timing_analysis = skip_timing_analysis,
		)
	else:
		raise RuntimeError(f'Dont know what tasks to run on run {repr(bureaucrat.run_name)} located in {bureaucrat.path_to_run_directory}')

if __name__=='__main__':
	import argparse
	
	grafica.plotly_utils.utils.set_my_template_as_default()
	
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
	parser.add_argument(
		'--perform-timing-analysis',
		help = 'Specify whether to perform a timing analysis too. This takes much longer time to calculate... Default is not to do it.',
		required = False,
		dest = 'timing_analysis',
		action = 'store_true'
	)
	args = parser.parse_args()
	
	Enrique = RunBureaucrat(Path(args.directory))
	main(
		bureaucrat = Enrique,
		force = args.force,
		skip_timing_analysis = not args.timing_analysis,
		number_of_processes = 3 if args.timing_analysis else max(multiprocessing.cpu_count()-1,1),
	)

