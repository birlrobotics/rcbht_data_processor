Primitives = ['bpos','mpos','spos','bneg','mneg','sneg','cons','pimp','nimp','none']
actionLbl  = ['a','i','d','k','pc','nc','c','u','n','z']
llbehLbl   = ['FX', 'CT', 'PS', 'PL', 'AL', 'SH', 'U', 'N']

label_mapping_dict = {}

label_mapping_dict['primitive'] = {}
for idx, val in enumerate(Primitives):
	label_mapping_dict['primitive'][val] = str(idx)

label_mapping_dict['primitive']['const'] = label_mapping_dict['primitive']['cons']

label_mapping_dict['composite'] = {}
for idx, val in enumerate(actionLbl):
	label_mapping_dict['composite'][val] = str(idx)

label_mapping_dict['llbehavior'] = {}
for idx, val in enumerate(llbehLbl):
	label_mapping_dict['llbehavior'][val] = str(idx)
