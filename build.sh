mkdir -p build/

set +x

g++ -std=c++23 src/main.cpp -o build/main -lm
