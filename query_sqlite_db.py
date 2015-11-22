import json
import sqlite3
import time

start_time = time.clock()

full_dataset_conn = sqlite3.connect('database.sqlite')
full_dataset_curs = full_dataset_conn.cursor()

controversial_dataset_conn = sqlite3.connect('uncontroversial_comments.sqlite')
controversial_dataset_curs = controversial_dataset_conn.cursor()

fields = ','.join('?' for _ in range(22))

num = 0
for row in full_dataset_curs.execute('SELECT * FROM May2015 WHERE controversiality=0 AND random() % 5 = 0 LIMIT 50000'):
  num += 1
  if num % 10000 == 0:
    print "Finished processing %d comments" % num

  controversial_dataset_curs.execute("INSERT INTO May2015 VALUES (%s)" % fields, row)
  controversial_dataset_conn.commit()

full_dataset_conn.close()
controversial_dataset_conn.close()

elapsed_time = time.clock() - start_time

print "Time elapsed: {} seconds".format(elapsed_time)
