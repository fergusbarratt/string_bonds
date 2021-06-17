from xmps.iMPS import iMPS
import numpy as np
import qmps.tools as tools
np.random.seed(500)

A = iMPS().random(2, 2).left_canonicalise()
B = iMPS().random(2, 2).left_canonicalise()
UA = tools.tensor_to_unitary(A)


