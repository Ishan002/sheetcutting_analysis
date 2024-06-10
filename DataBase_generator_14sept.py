import datetime
import random
import time
import mysql.connector as sql

class dbconnect():
    def __init__(self, hname, usr, pwd, datab, tablenm):
        self.db = sql.connect(
            host=hname,
            user=usr,
            password=pwd,
            database=datab,
            # auth_plugin='mysql_native_password'
        )
        if self.db.is_connected() == False:
            print("not connected")
        if self.db.is_connected() == True:
            print(" connected")
        self.curs = self.db.cursor()
        self.tablename = tablenm
      

    def add_dbdata(self, data):
        timestmp = str(datetime.datetime.now().strftime("%H:%M:%S"))
        print(data[0],data[1],data[2],data[3],data[4])
        datenm = str(datetime.datetime.today())
        try:
            self.curs.execute(f"INSERT INTO machine_monitoring(Location, date, time, machine, cam, machine_stat, duration) VALUES (%s, %s, %s, %s, %s, %s, %s)",(data[0], datenm, timestmp, data[1], data[2], data[3],data[4]))
            self.db.commit()
            print("Added data to database!")
        except:
            print("Data push failed... check parameters")

            


