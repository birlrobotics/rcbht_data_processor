def extract_features(folder_dict,dimension_dict):
	primitive_dict = folder_dict['primitive']
	composite_dict = folder_dict['composite']
	llbehavior_dict = folder_dict['llbehavior']

	keys = ['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz']

     # Use the maximum number of iterations that showed up for all axis (could also work with single axis)
	for key in keys:
		primitive_dict[key]  = slice_and_find_mode('primitive' , primitive_dict[key],  max(dimension_dict['primitive'].items(), key=lambda x: int(x[1]) )[1])
		composite_dict[key]  = slice_and_find_mode('composite',  composite_dict[key],  max(dimension_dict['composite'].items(), key=lambda x: int(x[1]) )[1])
		llbehavior_dict[key] = slice_and_find_mode('llbehavior', llbehavior_dict[key], max(dimension_dict['llbehavior'].items(),key=lambda x: int(x[1]) )[1])

def iteration_mapper(x):
	if 'Label' in x:
		return [x['t2End'], x['Label']]
	else:
		return [x['t2End'], x['CompLabel']]
  
'''Extract corresponding label, start and end time. Sort wrt time. Then, 
slice all the data into equal intervals to have same length'''
def slice_and_find_mode(level, dicts_cooked_from_iterations, slice):
	if slice == 0:
		return None

	if level == 'primitive':
		label_key      = 'Grad Label'
		end_time_key   = 'Finish'
		start_time_key = 'Start'
	elif level == 'composite':
		if 'Label' in dicts_cooked_from_iterations[0]:
			label_key = 'Label'
		else:
			label_key = 'CompLabel'
		end_time_key = 't2End'
		start_time_key = 't1Start'
	elif level == 'llbehavior':
		label_key      = 'CompLabel'
		end_time_key   = 't2End'
		start_time_key = 't1Start'
  
	#sort according to start time	
	dicts_cooked_from_iterations = sorted(dicts_cooked_from_iterations, key=lambda x:x[start_time_key])
     #Get start time of first iternation
	start_time = dicts_cooked_from_iterations[0][start_time_key]
     # Create a list of maps with the time at which each of the individual labels finishes
	iterations = map(lambda x:[x[end_time_key]-start_time, x[label_key]], dicts_cooked_from_iterations)

	result = []
     # Get the duration of the experiment for a specified RCBHT level
	time_length = iterations[-1][0]
	iteration_amount = len(iterations)
	
	s = 1
	slice_end = time_length*s/slice #slice an entire task in n slices
	count_dict = {}

	last_i_end = 0.0

	idx = 0
     ## Properly partition data
	while idx < iteration_amount:
		i = iterations[idx]
		if i[0] >= slice_end: # This time mark is greater than our slice, keep label and continue to slice 
			safe_add(i[1], slice_end-last_i_end, count_dict)

			result.append(get_mode(count_dict))

			count_dict = {}
			last_i_end = slice_end

			s += 1
			slice_end = time_length*(float(s)/slice) # Get the next slice limit
		else: #add label and time
			safe_add(i[1], i[0]-last_i_end, count_dict)
			last_i_end = i[0]
			idx += 1
 
	if len(result) != slice:
		print iterations
		print result
		print time_length
		print slice_end
		print s
		raise Exception

	return result


def safe_add(key, val, d):
	if key not in d:
		d[key] = val
	else:
		d[key] += val

def get_mode(d):
	mode_key = None
	mode_val = 0.0
	for key in d:
		if d[key] > mode_val:
			mode_key = key
			mode_val = d[key]

	if mode_key == None:
		print d

	return mode_key