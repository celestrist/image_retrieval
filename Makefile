TIME = `date "+ %H:%M:%S"`
LOGFILE = log/log.txt
LOGSTART = echo job:$@ start_time:$(TIME) >> $(LOGFILE)
LOGSTOP =  echo job:$@ stop_time:$(TIME) >> $(LOGFILE)
N = 10

all: extract mapper feature trainSet

extract:
	@$(LOGSTART)
	python src/extract_patch.py data/images/level1/ data/patches.csv --n_patches=200 --size=5
	@$(LOGSTOP)
mapper:
	@$(LOGSTART)
	python src/make_mapper.py data/patches.csv data/mapper.pkl --n_estimators=50	
	@$(LOGSTOP)
feature:
	@$(LOGSTART)
	python src/featurize.py data/images/level1/
	@$(LOGSTOP)
trainSet:
	@$(LOGSTART)
	python src/train_test.py data/images/level1/feature/
	@$(LOGSTOP)

	
