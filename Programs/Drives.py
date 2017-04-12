import numpy as np
from IO_utils import *
import perturb_field
import initialize
#import find_HII_bubbles

def drive_zscroll_noTs(ZHIGH, ZLOW, DZ):
	print "Now calling initialize"
	initialize.run()

	for z in np.arange(ZHIGH, ZLOW, DZ):
		print z
		perturb_field.run(z)
	return

if __name__=='__main__':
	drive_zscroll_noTs(16., 10., -2.)
