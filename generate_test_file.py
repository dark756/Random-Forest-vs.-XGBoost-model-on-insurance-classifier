import pandas as pd

df = pd.read_csv('AutoInsuranceClaims2024.csv')
test_df = df.sample(n=100, random_state=99) # Let's pull 100 rows for a better test size
test_df.to_csv('new_insurance_data_to_predict.csv', index=False)
print("Test data created WITH target column included.")