import datetime
import time
from scipy.stats import median_abs_deviation

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

def generate_distance_vs_n_position(df):
	"""Given a `df` that posses the columns x,y,z and n_positio
	
	Example input:
	```
														  x (m)     y (m)     z (m)
	n_waveform n_position n_trigger n_channel n_pulse                              
	0          0          0         1         1       -0.002926  0.001314  0.071339
	1          0          0         1         2       -0.002926  0.001314  0.071339
	2          0          0         4         1       -0.002926  0.001314  0.071339
	3          0          0         4         2       -0.002926  0.001314  0.071339
	4          0          1         1         1       -0.002926  0.001314  0.071339
	...                                                     ...       ...       ...
	148291     333        109       4         2       -0.002926  0.001647  0.071339
	148292     333        110       1         1       -0.002926  0.001647  0.071339
	148293     333        110       1         2       -0.002926  0.001647  0.071339
	148294     333        110       4         1       -0.002926  0.001647  0.071339
	148295     333        110       4         2       -0.002926  0.001647  0.071339
	```
	Example output:
	```
				Distance (m)
	n_position              
	0           0.000000e+00
	1           9.960938e-07
	2           2.001953e-06
	3           2.998047e-06
	4           4.003906e-06
	...                  ...
	329         3.289941e-04
	330         3.300000e-04
	331         3.309961e-04
	332         3.320020e-04
	333         3.329980e-04

	```
	"""
	if 'n_position' not in df.index.names:
		raise ValueError(f'`n_position` must be one index level of `df`.')
	df['Distance (m)'] = integrate_distance_given_path(list(df[['x (m)', 'y (m)', 'z (m)']].to_numpy()))
	distance = df.droplevel(list(set(df.index.names)-{'n_position'}),axis=0)['Distance (m)']
	distance = distance[~distance.index.duplicated(keep='first')]
	distance = distance.to_frame()
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
