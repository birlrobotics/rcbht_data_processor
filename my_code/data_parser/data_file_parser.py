def parse_file(file):
    iteration_texts = file.read().replace('\r', '').strip()

    iter_head = "Iteration"
    iteration_count = None

    dicts_cooked_from_iterations = []

    now_iter_start = iteration_texts.find(iter_head, 0)
    while now_iter_start != -1:
        next_iter_start = iteration_texts.find(iter_head, now_iter_start+len(iter_head))
        if next_iter_start == -1:
            iteration_text = iteration_texts[now_iter_start:]
        else:
            iteration_text = iteration_texts[now_iter_start: next_iter_start].strip()
        dict_cooked_from_iteration = mapper_text_to_dict(iteration_text)

        if iteration_count is None:
            iteration_count = dict_cooked_from_iteration['Iteration']
        
        if dict_cooked_from_iteration['Iteration'] == iteration_count: # Save the iteration info to the dictionary
            dicts_cooked_from_iterations.append(dict_cooked_from_iteration)
            iteration_count += 1

        now_iter_start = next_iter_start

    if len(dicts_cooked_from_iterations) == 0:
        print 'bad file with no axis data.'
        return None

    return purge_duplicate(dicts_cooked_from_iterations)


def mapper_text_to_dict(iteration_text):
    iteration_text = iteration_text.strip()

    dict_cooked_from_iteration = {}
    iteration_lines = iteration_text.split('\n')
    for i in iteration_lines:
        items = i.split(':')
        if len(items) != 2:
            continue
        key, val = items
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
