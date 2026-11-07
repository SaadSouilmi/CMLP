#include "nn.h"
#include "nn.cpp"

i32
main() {
	const char path_train[256] = "data/train.bin";
	const char path_test[256] = "data/test.bin";
	DataLoader loader;
	i32 status = load_file(path_train, &loader);
	status = load_file(path_test, &loader);

	close_loader(loader);
	return 0;
}
