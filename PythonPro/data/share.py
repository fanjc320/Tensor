try:
	import os
	import tushare as ts
except Exception:
	os.system('pip install tushare')
	
df = ts.get_hist_data('000001')
df.to_csv('000001.csv')