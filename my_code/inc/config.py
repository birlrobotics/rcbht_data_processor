states = ['approach','rotation','insertion','mating']
levels = ['primitive', 'composite', 'llbehavior']
axes = ['Fx','Fy','Fz','Mx','My','Mz']

endtimeKeyQueryDict = {
    "primitive" : "Finish",
    "composite" : "t2End",
    "llbehavior" : "t2End" 
}

starttimeKeyQueryDict = {
    "primitive" : "Start",
    "composite" : "t1Start",
    "llbehavior" : "t1Start" 
}


failure_x_base = ["+x", ""]
failure_y_base = ["+y", "-y", ""]
failure_r_base = ["+r", "-r", ""]

failure_class_name_to_id = {}
id_count = 0
for x in failure_x_base:
    for y in failure_y_base:
        for r in failure_r_base:
            now_class = x+y+r
            failure_class_name_to_id[now_class] = id_count
            id_count += 1 


