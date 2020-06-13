for ((i = 2; i <= 1024; i++));do
	python3 genMat.py $i > test_$i.txt
	time ../kernel.exe < test_$i.txt 
	rm test_$i.txt
done
