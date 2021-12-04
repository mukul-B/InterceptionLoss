import glob
import numpy as np
import pandas as pd
import pandasql as ps
import matplotlib.pyplot as plt
import seaborn as sns
import pydrive

#from pydrive.auth import GoogleAuth
#from pydrive.drive import GoogleDrive

#gauth = GoogleAuth()
#gauth.LocalWebserverAuth()

#drive = GoogleDrive(gauth)

#file1 = drive.CreateFile({'title':'Hello.txt'})
#file1.SetContentString('Hello World!')
#file1.Upload()


#file_list = drive.ListFile({'q': "'root' in parents and trashed=false"}).GetList()
#for file in file_list:
#    print('title: %s, id: %s'%(file['title'], file['id']))
