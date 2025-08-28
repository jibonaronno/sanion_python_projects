import sqlite3
from sqlite3 import Error
from os.path import join, dirname, abspath
from PyQt5.QtCore import QDateTime

dbfile = join(dirname(abspath(__file__)), 'flow.db')

class CRUD(object):
    def __init__(self, filename):
        self.filename = filename
        print(dbfile)

    def openDB(self, filename):
        conn = None
        try:
            conn = sqlite3.connect(filename)
            print("Database Connected.")
        except Error as e:
            print(e)
        return conn

    def openDBHard(self):
        self.con = self.openDB(dbfile)

    def insert_meter_data(self, data):
        #sql = '''INSERT INTO meter_data(datetime,content,devid) VALUES('11-30-2021 21:52:00', '12.343', '0x001')'''
        sql = '''INSERT INTO meter_data(datetime,content,devid) VALUES(?,?,?)'''
        cur = self.con.cursor()
        cur.execute(sql, data)
        self.con.commit()
        return cur.lastrowid

    def getListByDateRange(self, startd:QDateTime, endd:QDateTime, devid=None):
        #print(startd.date().toString("MM-dd-yyyy") + " " + startd.time().toString('HH:mm:ss'))
        #print(startd.toString("MM-dd-yyyy HH:mm:ss"))
        sql = ""
        data = []
        if devid:
            sql = "SELECT * FROM meter_data WHERE datetime BETWEEN '" + startd.toString("MM-dd-yyyy HH:mm:ss") + "' AND '" + endd.toString("MM-dd-yyyy HH:mm:ss") + "'" + "AND devid="+ str(devid)
        else:
            sql = "SELECT * FROM meter_data WHERE datetime BETWEEN '"+ startd.toString("MM-dd-yyyy HH:mm:ss") + "' AND '" + endd.toString("MM-dd-yyyy HH:mm:ss") + "'"
        cur = self.con.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        for row in rows:
            dtms = row[0].split(" ")
            data.append([dtms[0], dtms[1], row[2], row[1]])
        return data

    def insert_meter_data_hard(self):
        try:
            self.insert_meter_data(['11-30-2021 21:52:00', '12.343', '0x001'])
        except Error as e:
            print(e)

    def addRecord(self):
        pass
