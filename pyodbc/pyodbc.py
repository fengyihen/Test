# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 15:12:03 2016

@author: Yizhen
"""

import pyodbc

cnxn = pyodbc.connect('DRIVER={MySQL ODBC 5.1 Driver};SERVER=138.138.81.160;DATABASE=bigdatamodeldb;UID=root;PWD=123456')

cursor = cnxn.cursor()

cursor.execute("select * from  GTA_SEL2_INDEX_201612..SHL2_INDEX_000300_201612 where TRDDATE > 20161130 and TRDDATE < 20161202 order by TRDDATE,DATATIMESTAMP")

row = cursor.fetchall()

import pymysql

# 打开数据库连接
db = pymysql.connect("138.138.81.160","root","123456","bigdatamodeldb" )

# 使用 cursor() 方法创建一个游标对象 cursor
cursor = db.cursor()

# 使用 execute()  方法执行 SQL 查询 
cursor.execute("SELECT * FROM T_POSITION_2001")

for r in cursor.fetchall():

    print(r)
    
# 使用 fetchone() 方法获取单条数据.
data = cursor.fetchone()

print ("Database version : %s " % data)

# 关闭数据库连接
db.close()


