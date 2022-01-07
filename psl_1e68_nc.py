# As of Python 3.7, "Dict keeps insertion order" is the ruling.
# So please use Python 3.7 or later version.
import numpy as np
import sys
from psl import *

coordinates, charges = pqr_parser('1e68_model1.pqr')

filtration_idx = int(sys.argv[1])
p = float(sys.argv[2])
psl_l0, psl_l1 = [], []
psl = PSL(pts=coordinates, filtration_type = 'alpha', constant = False, charges=charges, scale = True)
psl.build_filtration()
psl.build_simplicial_pair(filtration_idx/100, p)
l0, l1 = psl.psl_0(), psl.psl_1() 
psl_l0.append(np.sort(np.linalg.eigvalsh(l0)))
psl_l1.append(np.sort(np.linalg.eigvalsh(l1)))

print('Calculation is done.')
    
mkdir_p('./1e68/nc_p={}'.format(p))
array_writer(psl_l0, './1e68/nc_p={}/psl_l0_nc_f={}.txt'.format(p, filtration_idx))
array_writer(psl_l1, './1e68/nc_p={}/psl_l1_nc_f={}.txt'.format(p, filtration_idx))

print('Files are written.')

