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

internal void
_matmul_mm(Matrix* C, const Matrix* A, const Matrix* B)
{
	for (u32 i = 0; i < A->m; ++i) {
		for (u32 k = 0; k < A->n; ++k) {
			for (u32 j = 0; j < B->n; ++j) {
				mat_at(C, i, j) += mat_at(A, i, k) * mat_at(B, k, j);
			}
		}
	}
}

internal void
_matmul_tm(Matrix* C, const Matrix* A, const Matrix* B)
{
	for (u32 k = 0; k < A->n; ++k) {
		for (u32 i = 0; i < A->m; ++i) {
			for (u32 j = 0; j < B->n; ++j) {
				mat_at(C, i, j) += mat_at(A, k, i) * mat_at(B, k, j);
			}
		}
	}
}

// TODO(Saad)
internal void
_matmul_mt(Matrix* C, const Matrix* A, const Matrix* B)
{

}

// TODO(Saad)
internal void
_matmul_tt(Matrix* C, const Matrix* A, const Matrix* B)
{

}
// TODO(Saad): handle transpose
// so add matmuls for mm tm mt tt
// Inspired by : https://www.youtube.com/watch?v=hL_n_GljC0I&t=1161s
internal i32
matmul(Matrix* C, const Matrix* A, const Matrix* B, i32 At, i32 Bt)
{
	u32 ARows = (At) : A->n ? A->m;
	u32 BRows = (Bt) : B->n ? B->m;
	u32 ACols = (At) : A->m ? A->n;
	u32 BCols = (Bt) : B->m ? B->n;
	if (ACols != BRows || C->m != ARows || C->n != BCols) { return -1; }
	
	matzero(C);
	switch((At<<1) | Bt)
	{
		case 0b00: { _matmul_mm(C, A, B); } break;
		case 0b01: { _matmul_mt(C, A, B); } break;
		case 0b10: { _matmul_tm(C, A, B); } break;
		case 0b00: { _matmul_tt(C, A, B); } break;
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

// TODO(Saad): To revisit, probably need to do this across rows/cols
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
