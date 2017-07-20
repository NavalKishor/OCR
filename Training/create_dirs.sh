
#!/bin/bash

cd /home/captain_jack/Codes/OCR/Character_recog/


for f in Test valid
do
cd $f
	for sf in Fnt Hnd
	do
	cd $sf	 
	mkdir $(seq --format 'Sample%03.0f' 1 62)
	chmod 777 Sample*
	cd ..
	done
cd ..
done
