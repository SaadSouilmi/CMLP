#import <stdlib.h>

#include "nn.h"
#include "viz.h"

#include "nn.cpp"

i32
main() {
	const char path_train[256] = "data/train.bin";
	const char path_test[256] = "data/test.bin";
	DataLoader loader;
	i32 status = load_file(path_train, &loader);
	
	u64 buffer_size = RecordSize*loader.count;
	u8* buffer = (u8*)malloc(buffer_size);
	f32* X     = (f32*)malloc(ImgSize*loader.count);
	u32* y     = (u32*)malloc(sizeof(u32)*loader.count);
	
	i32 n = dump_records(buffer, buffer_size, loader.count, &loader);
	read_img_label(X, y, n, buffer);
	
	u8 pixels[NFeatures];
	for (int i = 0; i < NFeatures; ++i) {
		pixels[i] = (u8)(X[i] * 255.0f + 0.5f);
	}

	show_grayscale(pixels, XDim, YDim);

	close_loader(&loader);
	return 0;
}
