make:
	g++ -O3 -DNDEBUG -o evaluate_object2 evaluate_object.cpp -g -std=c++17 -lstdc++fs
	g++ -O3 -DNDEBUG -o evaluate_bdd2 evaluate_bdd.cpp -g -std=c++17 -lstdc++fs

install:
	cp evaluate_object ~/bin

clean:
	rm evaluate_object2
	rm evaluate_bdd2
