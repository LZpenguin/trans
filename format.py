import pandas as pd

file = '/home/zbtrs/lz/trans/output/v2/testa_translate.csv'

df = pd.read_csv(file, encoding='utf-8')
df.to_csv(file.replace('.csv', '_w.csv'), encoding='utf-8-sig', index=False)