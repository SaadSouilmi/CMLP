mkdir -p build/

set +x

g++ -std=c++17 -fobjc-arc src/main.cpp src/viz.mm -framework Cocoa -o build/main
