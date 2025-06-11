import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Créer un DataFrame avec des valeurs NaN
df = pd.DataFrame({
    'a': [1, 2, np.nan, 4], 
    'b': [5, 6, 7, 8]
})

print('DataFrame original:')
print(df)

# Tester StandardScaler avec des valeurs NaN
scaler = StandardScaler()
try:
    transformed = scaler.fit_transform(df)
    print('\nAprès transformation:')
    print(transformed)
except Exception as e:
    print(f'\nErreur: {e}')
