import numpy as np
import pandas as pd

## Connectomes from Izhikevich paper,
# using neurons "nb1", "p23", "b", "nb", "ss4", "p4", "p5_p6", "TC", "TI", "TRN"

izh_matrix = pd.read_csv("normalized_weighted_data.csv", index_col=0)

# Local connectivity L1

n_izh_types = 10

# Shape [pre, post]
C_L1_local = np.zeros((n_izh_types, n_izh_types))

C_L1_local[0, 0] = izh_matrix["nb1"]["nb1"]  # nb1 -> nb1

# L1 -> L2/3 connections
C_L1_L2 = np.zeros((n_izh_types, n_izh_types))

C_L1_L2[0, 1] = izh_matrix["nb1"]["p23_1"]  # nb1 -> p23
C_L1_L2[0, 2] = izh_matrix["nb1"]["b23"]    # nb1 -> b
C_L1_L2[0, 3] = izh_matrix["nb1"]["nb23"]   # nb1 -> nb

# L2/3 -> L1 connections

C_L2_L1 = np.zeros((n_izh_types, n_izh_types))

C_L2_L1[1, 0] = izh_matrix["p23_23"]["nb1"]  # p23 -> nb1
C_L2_L1[2, 0] = izh_matrix["b23"]["nb1"]    # b -> nb1
C_L2_L1[3, 0] = izh_matrix["nb23"]["nb1"]   # nb -> nb1

# L2/3 -> L2 connections

C_L2_local = np.zeros((n_izh_types, n_izh_types))

C_L2_local[1, 1] = izh_matrix["p23_23"]["p23_23"]  # p23 -> p23
C_L2_local[1, 2] = izh_matrix["p23_23"]["b23"]    # p23 -> b
C_L2_local[1, 3] = izh_matrix["p23_23"]["nb23"]   # p23 -> nb
C_L2_local[2, 1] = izh_matrix["b23"]["p23_23"]    # b -> p23
C_L2_local[2, 2] = izh_matrix["b23"]["b23"]      # b -> b
C_L2_local[2, 3] = izh_matrix["b23"]["nb23"]     # b -> nb
C_L2_local[3, 1] = izh_matrix["nb23"]["p23_23"]  # nb -> p23
C_L2_local[3, 2] = izh_matrix["nb23"]["b23"]    # nb -> b
C_L2_local[3, 3] = izh_matrix["nb23"]["nb23"]   # nb -> nb



