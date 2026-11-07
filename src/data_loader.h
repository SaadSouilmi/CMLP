#ifndef DATALOADER_H
#define DATALOADER_H

#include <stdio.h>

#define XDim 28
#define YDim 28
#define NLabels 10
#define NFeatures 784
#define RecordSize 3140
#define ImgSize 3136

struct DataLoader {
	FILE* f;
	u32   count;
	u32   curr;
};

internal i32 load_file(const char* path, DataLoader* loader);
internal void close_loader(DataLoader* loader);

internal i32 dump_records(u8* buffer, u64 buffer_size, u64 n, DataLoader* loader);
internal void read_img_label(f32* X, u32* y, u64 n, const u8* buffer);


#endif // DATALOADER_H
