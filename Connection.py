import os
from impala.dbapi import connect
from impala.util import as_pandas
import pandas as pd

hive_hms_host = os.getenv(‘hive_hs2_host’, ‘Hive IP 입력’)
hive = connect(host = hive_hms_host, port = 10000, 
                                use_ssl=false, auth_mechanism = ‘ldap’,
                                # 계정/비밀번호 입력
                                user = ‘1234’, password=‘1234’)

sql = ‘select * from default.sample_07 limit 10’
hive_cursor = hive.cursor()
hive_cursor.execute(sql)

hive_pandas = as_pandas(hive_cursor)
hive_pandas.head(10)

hive_cursor.close()
hive.close()	
