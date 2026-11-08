#include <math.h>

internal i32
matcpy(Matrix* C, const Matrix* A) 
{
	if (C->m != A->m || C->n != A->n) { return -1; }
	
	memcpy(C->data, A->data, sizeof(f32) * (u64)C->m * C->n);
	return 0;
}

internal void
matzero(Matrix* C) 
{
	memset(C->data, 0, sizeof(f32) * (u64)C->m * C->n);	
}

internal f32
matmax(const Matrix* A)
{
	f32 c = A->data[0];
	u64 size = A->m * A->n;
	for (u64 i = 0; i < size; ++i) { c = Max(c, A->data[i]); }
	return c;
}

internal void
matscale(Matrix* C, f32 x) {
	u64 size = C->m * C->n;
	for (u64 i = 0; i < size; ++i) { C->data[i] *= x; }
}

internal void
matdiv(Matrix* C, f32 x) {
	u64 size = C->m * C->n;
	for (u64 i = 0; i < size; ++i) { C->data[i] /= x; }
}
	
internal i32
matadd(Matrix* C, const Matrix* A, const Matrix* B)
{
	if (A->n != B->n || C->n != A->n || A->m != B->m || C->m != A->m) { return -1; }
		
	for (u32 i = 0; i < A->m; ++i) {
		for (u32 j = 0; j < A->n; ++j) {
			mat_at(C, i, j) = mat_at(A, i, j) + mat_at(B, i, j);
		}
	}
	return 0;
}

internal i32
matsub(Matrix* C, const Matrix* A, const Matrix* B)
{
	if (A->n != B->n || C->n != A->n || A->m != B->m || C->m != A->m) { return -1; }
		
	for (u32 i = 0; i < A->m; ++i) {
		for (u32 j = 0; j < A->n; ++j) {
			mat_at(C, i, j) = mat_at(A, i, j) - mat_at(B, i, j);
		}
	}
	return 0;
}

internal i32
matmul(Matrix* C, const Matrix* A, const Matrix* B)
{
	if (A->n != B->m || C->m != A->m || C->n != B->n) { return -1; }
	
	matzero(C);
	for (u32 i = 0; i < A->m; ++i) {
		for (u32 k = 0; k < A->n; ++k) {
			for (u32 j = 0; j < B->n; ++j) {
				mat_at(C, i, j) += mat_at(A, i, k) * mat_at(B, k, j);
			}
		}
	}
	return 0;
}

internal i32
relu(Matrix* C, const Matrix* A)
{
	if (A->n != C->n || A->m != C->m) { return -1; }

	for (u32 i = 0; i < A->m; ++i) {
		for (u32 j = 0; j < A->n; ++j) {
			mat_at(C, i, j) = Max(mat_at(A, i, j), 0);
		}
	}
	return 0;
}

internal i32
softmax(Matrix* p, const Matrix* logits)
{
	if (p->n != logits->n || p->m != logits->m) { return -1; }

	f32 c = matmax(logits);
	f32 sum = 0;
	u64 size = p->n * p->m;
	for (u64 i = 0; i < size; ++i) {
		p->data[i] = exp(logits->data[i] - c);
		sum += p->data[i];
	}
	matdiv(p, sum);
	return 0;
}

internal i32
mse(f64* L, const Matrix* y_hat, const Matrix* y)
{
	if (y_hat->n != y->n || y_hat->m != y->m) { return -1; }
	
	f64 mean = 0;
	u64 size = y->m * y->n;
	for (u64 i = 0; i < size; ++i) {
		f64 l = (f64)(y_hat->data[i] - y->data[i]) * (y_hat->data[i] - y->data[i]);
		mean += (l - mean)/(i+1);
	}
	*L = mean;
	return 0;
}
