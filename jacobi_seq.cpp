#include <cstdio>
#include <math.h>
#include <string>
#include <cstring>
#include <sys/time.h>
using namespace std;



// Return current wallclock time, for performance measurement
uint64_t GetTimeStamp() {
    struct timeval tv;
    gettimeofday(&tv,NULL);
    return tv.tv_sec*(uint64_t)1000000+tv.tv_usec;
}


void mat_multiply(const int n_row, const int n_col, const double* A, const double* x, double* y){

    for(int i = 0 ; i < n_row; ++i){
        y[i] = 0;
        for (int j = 0 ; j < n_col ; ++j){
            y[i] = y[i] + A[n_row*i + j]*x[j];
        }
    }
}


void seq_jacobi(const int grid_size, const double* A, double* b, double* x, int max_iter=10000, double l2_termination=1e-6){

    const int n = grid_size*grid_size;
    double *D = new double[n];
    double *y = new double[n];

    for(int i = 0 ; i < n; ++i){
        for(int j = 0 ; j < n; ++j){
            if(i==j)
                D[i] = A[n*i + j];
        }
    }

    double l2_norm = l2_termination + 1;
    int itr = 0;

    mat_multiply(n, n, A, x, y);

    while((itr < max_iter) && (l2_norm > l2_termination)){
        for(int i = 0 ; i < n; ++i){
            x[i] = (b[i] - (y[i] - D[i]*x[i]))/D[i];
        }

        mat_multiply(n, n, A, x, y);

        l2_norm=0;
        for (int i = 0;i<n;i++){
            l2_norm = l2_norm + pow(b[i]-y[i],2);
        }

        l2_norm = sqrt(l2_norm);
        itr++;
    }

}

void gen_matrix(const int grid_size, double* A, bool display=false){

	const int n_row = grid_size*grid_size;
	const int n_col = grid_size*grid_size;

    for(int i = 0 ; i < n_row; ++i){
        for(int j = 0 ; j < n_col; ++j){
            if(i == j){
                A[n_row*i+j] = 4;
                if(i%grid_size ==0){
                    for(int r = 0 ; r < grid_size; ++r){
                        for(int c = 0 ; c < grid_size; ++c){
                            if(r == c+1 || c == r+1)
                                A[n_row*(r+i)+(c+j)] = -1;
                        }
                    }
                }
            }else if ( i == j + grid_size || j == i + grid_size){
                A[n_row*i+j] = -1;
            }
        }
    }

	if(display){
        char sep ='\0';
		for(int i = 0 ; i < n_row; ++i){
			for(int j = 0 ; j < n_col; ++j){
				printf("%c\t%.1f", sep, A[i*n_col+j]);
				sep = ',';
			}
			printf("\n");
			sep ='\0';
		}
        printf("\n");
	}
}

void gen_b(const int grid_size, double* b, bool display=false){

    const int size = grid_size*grid_size;
    for(int idx = 0; idx < size; ++idx){
        b[idx] = 0;
    }

    for(int idx = 0 ; idx < size; ++idx){
        int row=(idx/grid_size)+1, col=(idx%grid_size)+1;
        if(col==1)
            b[idx] += sin(((double)row*1.0)/(double)grid_size);
        else if(col == grid_size)
            b[idx] += 0.0;
        
        if(row == 1)
            b[idx] += ((double)col*1.0)/grid_size;
        else if (row == grid_size)
            b[idx] += 0.0;
    }

    if(display){
        for(int i = 0 ; i < size; ++i){
            printf("%.3f ", b[i]);
        }
        printf("\n");
    }
}

void show_vec(double* x, const int grid_size){
    for(int i = 0 ; i < grid_size*grid_size; ++i){
        printf("%.3f ", x[i]);
    }
    printf("\n");
}


int main(int argc, char* argv[])
{
    // g++ jacobi_seq.cpp -o jacobi_seq && ./jacobi_seq 4   
    const int grid_size = atoi(argv[1]);
    const int n_rows = grid_size*grid_size;
    const int n_cols = grid_size*grid_size;

    double* A = new double[n_rows*n_cols]{0};
    gen_matrix(grid_size, A);

    double* b = new double[n_cols];
    gen_b(grid_size, b);

    double* x = new double[n_rows];

    uint64_t start = GetTimeStamp();
    seq_jacobi(grid_size, A, b, x);
	printf("Time: %ld us\n", (uint64_t) (GetTimeStamp() - start));

    // printf("\n");
    // show_vec(x, grid_size);



    // const int n = grid_size*grid_size;
    // double x[n] = {0};
    // int max_iter = 1e+6;

    // uint64_t start = GetTimeStamp();
    // jacobi_iter(n, A, b, x, max_iter);
	// print the time taken to do the computation
	// printf("Time: %ld us\n", (uint64_t) (GetTimeStamp() - start));



    return 0;
}