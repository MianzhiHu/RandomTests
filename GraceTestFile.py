import numpy as np
import pandas as pd

x = np.log(2)
print(x)

y = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
print(y)

# turn x into a dataframe
x = pd.DataFrame({'log_2': [x]})

# save x and y
x.to_csv('test_output_x.csv', index=False)
y.to_csv('test_output_y.csv', index=False)