'''Encode string labels for the three layers of RCBHT with an alphabet long enough that it can encode with a single character. To this end we choose the a-z alphabet to encode 24 different labels accross the the 3 RCBHT levels.'''

#------------------------------------------------------------------------
# gradLbl2gradInt
# -Imp -Big -Med -Small Const Small Med Big Imp
#   -4   -3   -2     -1   0       1   2   3   4
#------------------------------------------------------------------------
def encodeRCBHTList(s,level):
    seqLen=len(s)

    for i in range(seqLen):
        if level=='primitive':
            gradLbl2abc(s,i)
        elif level=='composite':
            actionLbl2abc(s,i)
        elif level=='llbehavior':
            behLbl2abc(s,i)


def gradLbl2abc(dLabel,i):

    if dLabel[i]== 'bpos':
        dLabel[i] = 'a'
    elif dLabel[i]== 'mpos':
        dLabel[i] = 'b'
    elif dLabel[i]== 'spos':
        dLabel[i] = 'c'
    elif dLabel[i]== 'bneg':
        dLabel[i] = 'd'
    elif dLabel[i]== 'mneg':
        dLabel[i] = 'e'
    elif dLabel[i]== 'sneg':
        dLabel[i] = 'f'
    elif dLabel[i]== 'cons':
        dLabel[i] = 'g'
    elif dLabel[i]== 'pimp':
        dLabel[i] = 'h'
    elif dLabel[i]== 'nimp':
        dLabel[i] = 'i'
    elif dLabel[i]== 'none':
        dLabel[i] = 'j'

##-------------------------------------------------------------------------
# actionInt2actionLbl[i]
# a i d k pc nc c u
# 1 2 3 4 5  6  7 8
##-------------------------------------------------------------------------
def actionLbl2abc(actionLbl,i):

    # Convert labels to ints
    if actionLbl[i]=='a':
        actionLbl[i] = 'k'    	# alignment
    elif actionLbl[i]== 'i':
        actionLbl[i] = 'l'    	# increase
    elif actionLbl[i]== 'd':
        actionLbl[i] = 'm'    	# decrease
    elif actionLbl[i]== 'k':
        actionLbl[i] = 'n'    	# constant
    elif actionLbl[i]== 'pc':
        actionLbl[i] = 'o'    	# positive contact
    elif actionLbl[i]== 'nc':
        actionLbl[i] = 'p'    	# negative contact
    elif actionLbl[i]== 'c':
        actionLbl[i] = 'q'   	# contact
    elif actionLbl[i]== 'u':
        actionLbl[i] = 'r'    	# unstable

#-------------------------------------------------------------------------
#   llbehLbl   = {'FX' 'CT' 'PS' 'PL' 'AL' 'SH' 'U' 'N');   
#                {'fix' 'cont' 'push' 'pull' 'align' 'shift' 'unstable' 'noise');
#   llbehLbl    = [ 1,   2,   3,   4,   5,   6,   7,  8];
##-------------------------------------------------------------------------
def behLbl2abc(llbLabel,i):

    # Convert labels to ints
    if llbLabel[i] == 'FX':
        llbLabel[i] = 's'    	# Fixed
    elif llbLabel[i]== 'CT':
        llbLabel[i] = 't'    	# contact
    elif llbLabel[i]== 'PS':
        llbLabel[i] = 'u'    	# push
    elif llbLabel[i]== 'PL':
        llbLabel[i] = 'v'    	# pull
    elif llbLabel[i]== 'AL':
        llbLabel[i] = 'w'    	# alignment
    elif llbLabel[i]== 'SH':
        llbLabel[i] = 'x'    	# shift
    elif llbLabel[i]== 'U':
        llbLabel[i] = 'y'; 	# unstable
    elif llbLabel[i]== 'N':
        llbLabel[i] = 'z'    	# none
