#include <string.h>


internal i32 
load_file(const char* path, DataLoader* loader) 
{
	FILE* f = fopen(path, "rb");
	if (!f) { return -1; }
	
	u8 header[16];
	const u8* p = header;
	if (fread(header, 1, 16, f) != 16) { return -1; }

	char name[5];
	u32 version;
	u32 n_samples;
	u32 n_features;
	
	memcpy(name, p, 4);
	name[4] = '\0';
	p += 4;
	memcpy(&version, p, 4);
	p += 4;
	memcpy(&n_samples, p, 4);
	p += 4;
	memcpy(&n_features, p, 4);

	printf("%s: version = %d, n_samples = %d, n_features = %d\n", name, version, n_samples, n_features);

	loader->f = f;
	loader->count = n_samples;
	loader->curr  = 0;
	
	return 0;
}

internal void
close_loader(DataLoader* loader)
{
	if (loader->f) { fclose(loader->f); }
}

internal i32
dump_records(u8* buffer, u64 buffer_size, u64 n, DataLoader* loader)
{
	if (loader->curr >= loader->count) { return 0; }

	n = Min(n, buffer_size/RecordSize);	
	n = Min(n, loader->count - loader->curr);
	
	u64 read = fread(buffer, RecordSize, n, loader->f);
	loader->curr += read;
	return read;
}

internal void
read_img_label(f32* X, u32* y, u64 n, const u8* buffer) {
	for (u64 i = 0; i < n; ++i) {
		memcpy(X, buffer, ImgSize);
		X += NFeatures;
		buffer += ImgSize;
		
		memcpy(y, buffer, 4);
		buffer += 4;
		++y;
	}
}
