TIME = `date "+ %H:%M:%S"`
LOGFILE = log/log.txt
LOGSTART = echo job:$@ start_time:$(TIME) >> $(LOGFILE)
LOGSTOP =  echo job:$@ stop_time:$(TIME) >> $(LOGFILE)
N = 10

all: 

extract:
	python src/extract_patch.py data/images/level1/ data/patches.csv --n_patches=100 --size=6
