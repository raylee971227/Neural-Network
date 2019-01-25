run.exe: main.o NeuralNet.o
	g++ -std=c++11 -o run.exe main.o NeuralNet.o
main.o: main.cpp
	g++ -std=c++11 -c main.cpp
NeuralNet.o: NeuralNet.cpp NeuralNet.h
	g++ -std=c++11 -c NeuralNet.cpp
debug:
	g++ -g -o debug.exe main.cpp NeuralNet.cpp
clean:
	rm -f *.exe *.o *.stackdump *~
run:
	./run.exe
