import setup as s
import weights as w
import pandas as pd
import numpy as np

f = s.init_feature_matrix()
lab = s.init_label_vec()
we = s.init_theta()
o = w.gradient_descent(f, lab, we, .0001, 200)

total = 0
for vec, label in zip(f, lab):
    p = w.predict(vec, o)
    total = total + 1 if p == label else total
print(total / f.shape[0])