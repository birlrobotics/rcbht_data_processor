import inc.config as config

'''
@param dict_of_one_experiment
    this should be a dictionary whose structure should be like:
        {
            level : { 
                axis : [ 
                    {
                       # key-value pairs of one iteration
                    },
                    ... # more iterations
                ]
                ... # more axes
            }
            ... # more levels
        }
'''

streaming_interval = 0.5#second

def print_and_verify_things(dict_of_one_experiment, array_of_streaming_dicts):
    level_to_check = config.levels[2]
    axis_to_check = config.axes[2]

    print '-'*20
    print [i[config.endtimeKeyQueryDict[level_to_check]] for i in dict_of_one_experiment[level_to_check][axis_to_check]]
    print '*'*5
    for one_stream_dict in array_of_streaming_dicts:
        print "&&"
        print [i[config.endtimeKeyQueryDict[level_to_check]] for i in one_stream_dict[level_to_check][axis_to_check]]
    raw_input()

def stream_one_experiment(dict_of_one_experiment):
    dict_for_streaming_axes = {}

    streams_count = -1 

    timeLengthOfThisExp = -1;
    for level in config.levels:
        dict_for_streaming_axes[level] = {}

        for axis in config.axes:
            dict_for_streaming_axes[level][axis] = []

            iteration_amount = len(dict_of_one_experiment[level][axis])
            idx = 0

            first_iteration = dict_of_one_experiment[level][axis][0]
            last_end_time = first_iteration[config.starttimeKeyQueryDict[level]]
            while idx < iteration_amount:
                now_iteration = dict_of_one_experiment[level][axis][idx]
                now_end_time = now_iteration[config.endtimeKeyQueryDict[level]]

                if now_end_time < last_end_time+streaming_interval:
                    idx += 1
                    if idx == iteration_amount:
                        import copy
                        new_streaming_axis = copy.deepcopy(dict_of_one_experiment[level][axis][:idx])
                        dict_for_streaming_axes[level][axis].append(new_streaming_axis)
                    continue
                elif now_end_time == last_end_time+streaming_interval:
                    import copy
                    new_streaming_axis = copy.deepcopy(dict_of_one_experiment[level][axis][:idx+1])
                    dict_for_streaming_axes[level][axis].append(new_streaming_axis)
                    
                    idx += 1
                    last_end_time += streaming_interval 
                else:
                    import copy
                    new_streaming_axis = copy.deepcopy(dict_of_one_experiment[level][axis][:idx+1])
                    new_streaming_axis[-1][config.endtimeKeyQueryDict[level]] = last_end_time+streaming_interval
                    dict_for_streaming_axes[level][axis].append(new_streaming_axis)
                    last_end_time += streaming_interval 
            if streams_count == -1: 
                streams_count = len(dict_for_streaming_axes[level][axis])
            elif len(dict_for_streaming_axes[level][axis]) != streams_count:
                print "a bad exp with different streams count"
                return None
    

    array_of_streaming_dicts = []

    for i in range(streams_count):
        now_streaming_dict = {}
        for level in config.levels:
            now_streaming_dict[level] = {}
            for axis in config.axes:
                now_streaming_dict[level][axis] = dict_for_streaming_axes[level][axis][i]
        array_of_streaming_dicts.append(now_streaming_dict)

    #print_and_verify_things(dict_of_one_experiment, array_of_streaming_dicts)
    return array_of_streaming_dicts
