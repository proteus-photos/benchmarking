import numpy as np

SIZE   = 100
FILTER = 3
image  = np.random.random((SIZE, SIZE))

filter  = np.ones((FILTER, FILTER))
act_map = np.zeros((SIZE-FILTER+1, SIZE-FILTER+1))

for i in range(SIZE-FILTER+1):
    for j in range(SIZE-FILTER+1):
        act_map[i][j] = np.sum(image[i:FILTER+i,j:FILTER+j]*filter)