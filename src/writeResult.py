import pandas as pd
import os
from collections import OrderedDict
from subprocess import check_call
from shutil import copyfile

def do(result,test):
	# count number of files
	path, dirs, files = os.walk("../results").next()
	file_count = len(files)/2+1

	# Write the test results
	data=OrderedDict()
	data["id"]=test["id"] 
	data["EAP"]=result["EAP"]
	data["HPL"]=result["HPL"]	
	data["MWS"]=result["MWS"]
	output = pd.DataFrame(data=data)
	filename = "../results/result"+str(file_count)+".csv"
	output.to_csv( filename, index=False )
	filename = "../results/result"+str(file_count)+"compr.csv"
	output.to_csv( filename, index=False )
	check_call(['gzip', filename])
