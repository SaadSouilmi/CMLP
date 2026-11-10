#import <stdlib.h>
#import <math.h>

#include "nn.h"
#ifdef LOADVIZ
#include "viz.h"
#endif // LOADVIZ

#include "nn.cpp"

internal void print_img(const f32* pixels);
internal void print_hist(const Matrix* p);

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
	


	print_img(X);
	f32 probs[10] = {
		0.005f, 0.010f, 0.025f, 0.040f, 0.015f,
		0.050f, 0.005f, 0.735f, 0.080f, 0.035f
	};			
	Matrix p = {.m=1, .n=10, .data=probs};
	print_hist(&p);

	close_loader(&loader);
	free(buffer); free(X); free(y);
	return 0;
}

internal void
print_img(const f32* pixels)
{
	for (u32 i = 0; i < XDim; i+=2) {
		for (u32 j = 0; j < YDim; ++j) {
			u32 top = pixels[(u64)i * YDim + j] * 255.0f + 0.5f;
			u32 bot = pixels[(u64)(i+1) * YDim + j] * 255.0f + 0.5f;
			printf("\x1b[38;2;%u;%u;%u;48;2;%u;%u;%um▀",
               top, top, top, bot, bot, bot);
		}
		printf("\n");
	}
}

internal void print_hist(const Matrix* p) 
{
    constexpr int height = 5;
    constexpr int steps = height * 8;
    const char* blocks[] = {" ", "▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"};

    size_t count = (size_t)p->m * p->n;
    if (count == 0) return;

    // Wide enough for the largest index printed below the bars.
    int width = snprintf(nullptr, 0, "%zu", count - 1);

    printf("\x1b[97m");  // bright white

    for (int row = height - 1; row >= 0; --row) {
        for (size_t i = 0; i < count; ++i) {
            int filled = (int)(p->data[i] * steps + 0.5f);
            int level = filled - row * 8;
            if (level < 0) level = 0;
            if (level > 8) level = 8;

            for (int k = 0; k < width; ++k)
                fputs(blocks[level], stdout);
            putchar(' ');
        }
        putchar('\n');
    }

    printf("\x1b[39m");  // restore default color for labels
    for (size_t i = 0; i < count; ++i)
        printf("%*zu ", width, i);
    putchar('\n');
}
