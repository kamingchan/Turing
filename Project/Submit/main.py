import os
from ftplib import FTP
from zipfile import ZipFile
import time

group = '084'
current_time = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
version = input("Please input version\n")
zip_filename = '%s_result_v%s.zip' % (group, version)
files = list()
for x in range(1, 5):
    old_name = '%d.txt' % x
    if os.path.isfile(old_name):
        new_name = '%s_%d_v%s.txt' % (group, x, version)
        os.rename(old_name, new_name)
        files.append(new_name)

with ZipFile(zip_filename, 'w') as f:
    for txt in files:
        f.write(txt)
        os.remove(txt)

# ftp = FTP('my.ss.sysu.edu.cn')
# ftp.encoding = 'utf-8'
# ftp.login()
# ftp.cwd('~ryh/se46000089_2016_fall/Project')
# with open(zip_filename) as f:
#     ftp.storbinary('STOR %s' % (zip_filename,), f)
# ftp.dir()
# ftp.close()
# os.rename(zip_filename, '%s.zip' % current_time)
