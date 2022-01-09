#include <stdio.h>

int c_dot(const int *a, const int *b, int len) {

	int c = 0;
	for (int i=0; i<len; i++){
		c += a[i]*b[i];
	}
	return c;

}

int c_sum(const int *a, int len) {

	int c = 0;
	for (int i=0; i<len; i++){
		c += a[i];
	}
	return c;
}

void speed_test(){
	int size = 10000;
	int a[size];
	for (int i = 0; i<size; i++){
		for (int j = 0; j<size; j++)
			for (int k = 0; k<size; k++)
				a[i] = i+j+k;
	}
}

int main(){
	speed_test();
}
