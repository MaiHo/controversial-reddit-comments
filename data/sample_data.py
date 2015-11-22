import json
import sqlite3
import time

start_time = time.clock()

full_dataset_conn = sqlite3.connect('database.sqlite')
full_dataset_curs = full_dataset_conn.cursor()

sample_dataset_conn = sqlite3.connect('sample_comments.sqlite')
sample_dataset_curs = sample_dataset_conn.cursor()

fields = ','.join('?' for _ in range(22))

num = 0
# Change this query according to how you want to sample the data.
for row in full_dataset_curs.execute('SELECT * FROM May2015 WHERE 1=1'):
  num += 1
  if num % 10000 == 0:
    print "Finished processing %d comments" % num

  sample_dataset_curs.execute("INSERT INTO May2015 VALUES (%s)" % fields, row)
  sample_dataset_conn.commit()

full_dataset_conn.close()
sample_dataset_conn.close()

elapsed_time = time.clock() - start_time

print "Time elapsed: {} seconds".format(elapsed_time)
