import pandas as pd

people = {"first": ['Corey', 'Demet', 'Dogukan'],
          "last": ['Schafer', 'Demir', 'Demir'],
          "email": ['Corey.Schafer@email.com', 'Demet.Demir@email.com', 'Dogukan.Demir@email.com']
        }

df = pd.DataFrame(people)
print(df)
print("")
print(df.iloc[0:, 0:1])
print(df.iloc[0:, 2])
print(df.loc[['email']])