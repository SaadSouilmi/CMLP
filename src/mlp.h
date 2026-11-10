#ifndef MLP_H
#define MLP_H

#include <stdlib.h>

struct Matrix {
	u32 m; u32 n;
	f32* data;
};
#define mat_at(A, i, j) ((A)->data[(u64)(i)*(A)->n + (j)])


internal i32 matcpy(Matrix* C, const Matrix* A);
internal void matzero(Matrix* C);
internal f32 matmax(const Matrix* A);

internal void matscale(Matrix* C, f32 x);
internal void matdiv(Matrix* C, f32 x);
internal i32 matadd(Matrix* C, const Matrix* A, const Matrix* B);
internal i32 matsub(Matrix* C, const Matrix* A, const Matrix* B);
internal i32 matmul(Matrix* C, const Matrix* A, const Matrix* B);

internal i32 relu(Matrix* C, const Matrix* A);
// TODO(Saad): see mlp.cpp
internal i32 softmax(Matrix* p, const Matrix* logits);

#endif // MLP_H
