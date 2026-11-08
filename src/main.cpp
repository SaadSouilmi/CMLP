#import <stdlib.h>
#import <math.h>

#include "nn.h"
#ifndef LOADVIZ
#include "viz.h"
#endif // LOADVIZ

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
	
	f64 mean = 0.0;
	f64 m2   = 0.0;
	f64 std  = 0.0;
	u64 count = 0;
	f32 min = X[0];
	f32 max = X[0];
	for (u64 i = 0; i < n*NFeatures; ++i)
	{
		f32 x = X[i];
		min = Min(min, x);
		max = Max(max, x);

		++count;
		f64 delta = x - mean;
		mean += delta/(f64)count;
		m2   += delta * (x - mean);
	}
	std = sqrt(m2 / count);

	printf("Mean = %f, Std = %f, Min = %f, Max = %f\n", mean, std, min, max);

	close_loader(&loader);
	return 0;
}
