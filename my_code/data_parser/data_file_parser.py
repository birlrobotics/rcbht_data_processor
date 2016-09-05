def parse_file(file):
	iteration_texts = file.read().replace('\r', '').strip().split('\n\n') # split data by carriage return and by two new lines

	dicts_cooked_from_iterations = []
     # Now strip each individual "iteration" block from file
	iteration_count = 1
	for iteration_text in iteration_texts:
		iteration_text = iteration_text.strip() # 
		if iteration_text == '':
			continue
		dict_cooked_from_iteration = mapper_text_to_dict(iteration_text)
		if dict_cooked_from_iteration['Iteration'] == iteration_count: # Save the iteration info to the dictionary
			dicts_cooked_from_iterations.append(dict_cooked_from_iteration)
			iteration_count += 1

	if len(dicts_cooked_from_iterations) == 0:
		print 'bad file with no axis data.'
		return None

	return purge_duplicate(dicts_cooked_from_iterations)


def mapper_text_to_dict(iteration_text):
	iteration_text = iteration_text.strip()

	dict_cooked_from_iteration = {}
	iteration_lines = iteration_text.split('\n')
	for i in iteration_lines:
		key, val = i.split(':')


		key = key.strip()
		val = val.strip()
		try :
			val = float(val)
		except ValueError:
			pass
		dict_cooked_from_iteration[key] = val
	return dict_cooked_from_iteration


def purge_duplicate(dicts_cooked_from_iterations):
	s = {}
	for i in dicts_cooked_from_iterations:
		if i['Iteration'] not in s:
			s[i['Iteration']] = i

	return s.values() 
