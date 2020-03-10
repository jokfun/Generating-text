import re
import sys

if len(sys.argv)==2:
	print("Clearing ",sys.argv[1])
	f = open(sys.argv[1])
	text = f.read()
	f.close()
	regex = re.compile('[^a-zA-Z0-9 !.,?;"-():/\n]')
	rs = regex.sub('',text)
	name = sys.argv[1].split(".")[0]
	f = open(name+"_copy.txt","w")
	f.write(rs)
	f.close()
