#include <math.h>
#include <mpi.h>
#include <assert.h>

// The following code comes from https://github.com/ajdsouza/Parallel-MPI-Jacobi.git
// use MPI to solve the linear equation Ax = b


void get_grid_comm(MPI_Comm* grid_comm)
{
    // get comm size and rank
    int rank, p;

    // Get the number of processes of the world
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    // Get the rank of a process in the communicator
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);


    int q = (int)sqrt(p);

    // split into grid communicator
    int dims[2] = {q, q};
    int periods[2] = {0, 0};
    
    MPI_Cart_create(MPI_COMM_WORLD, 2, 
                        dims, periods, 
                        0, grid_comm);
    // from page 51, mpi
    // MPI_Cart_create(comm_old, ndims, dims[], periods[], reorder, * comm_cart)
    // Input Parameters
    //      comm_old : input communicator (handle)
    //      ndims : number of dimensions of cartesian grid (integer)
    //      dims : integer array of size ndims specifying the number of processes in each dimension
    //      periods : logical array of size ndims specifying whether the grid is periodic (true) or not (false) in each dimension
    //      reorder : ranking may be reordered (true) or not (false) (logical)
    // Output
    //      comm_cart : communicator with new cartesian topology (handle)

}

int block_decompose(const int n, const int p, const int rank)
{
    return n / p + ((rank < n % p) ? 1 : 0);
}

int block_decompose(const int n, MPI_Comm comm)
{
    int rank, p;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &p);
    return block_decompose(n, p, rank);
}

int block_decompose_by_dim(const int n, MPI_Comm comm, int dim)
{
    // get dimensions
    int dims[2];
    int periods[2];
    int coords[2];
    MPI_Cart_get(comm, 2, dims, periods, coords);
    return block_decompose(n, dims[dim], coords[dim]);
}



void distribute_vector(const int n, double* input_vector, double** local_vector, MPI_Comm comm)
{
	// TODO
    //std::cout<<"DEBUG:mpi_jacobi::distribute_vector:begin with n="<<n<<std::endl;

    // rank in grid
    int cur_rank;
    int cur_coords[2];
    MPI_Comm_rank(comm, &cur_rank);
    // i,j coord in grid
    MPI_Cart_coords(comm, cur_rank, 2, cur_coords);

    // rank of root 0,0 matrix in grid
    int rank_root;
    int coords_root[2]={0,0};
    MPI_Cart_rank(comm, coords_root, &rank_root);

    // recieved input to distribute
    //if ( cur_rank == rank_root ) {
    //for (int i=0;i<n;i++){
        //std::cout<<"DEBUG:mpi_jacobi::distribute_vector:(i,j)=("<<cur_coords[0]<<","<<cur_coords[1]<<"),recieved input_vector["<<i<<"]="<<input_vector[i]<<std::endl;
    //}
    //}

    // create a comm group for each column
    MPI_Comm comm_col;
    int cdims[] = {1,0};
    MPI_Cart_sub(comm, cdims, &comm_col);

    //std::cout<<"DEBUG:mpi_jacobi::distribute_vector:(i,j)=("<<cur_coords[0]<<","<<cur_coords[1]<<"),created cart sub for rank="<<cur_rank<<std::endl;

    // distribute vector among first column processors only 
    if ( cur_coords[1]==0 ){
		//std::cout<<"DEBUG:mpi_jacobi::distribute_vector:(i,j)=("<<cur_coords[0]<<","<<cur_coords[1]<<"),rank="<<cur_rank<<std::endl;
      	// rank in the column
		int col_size;
		int cur_col_rank;
		MPI_Comm_size(comm_col, &col_size);
		MPI_Comm_rank(comm_col, &cur_col_rank);

      	//std::cout<<"DEBUG:mpi_jacobi::distribute_vector:(i,j)=("<<cur_coords[0]<<","<<cur_coords[1]<<"),n="<<n<<",colrank="<<cur_col_rank<<",col_size="<<col_size<<std::endl;

      	// floor and ceil values
		int fnp = (int)floor(((double)n)/col_size);
		int cnp = (int)ceil(((double)n)/col_size);

      	//std::cout<<"DEBUG:mpi_jacobi::distribute_vector:(i,j)=("<<cur_coords[0]<<","<<cur_coords[1]<<",cnp="<<cnp<<",fnp="<<fnp<<std::endl;

      	// allocate memory for rcv buffer
		int rcv_size = cur_col_rank < ( n % col_size ) ? cnp : fnp;

      	//std::cout<<"DEBUG:mpi_jacobi::distribute_vector:(i,j)=("<<cur_coords[0]<<","<<cur_coords[1]<<"),Allocation memory for local vector="<<rcv_size<<std::endl;

      	double *tmp_vec;
	  	tmp_vec=(double *)malloc(rcv_size*sizeof(double));
		*local_vector = &tmp_vec[0];

      	// rank of root 0,0 matrix in the col comm to be used in scatterv
		int rank_col_root;
		MPI_Cart_rank(comm_col,coords_root,&rank_col_root);

      	//std::cout<<"DEBUG:mpi_jacobi::distribute_vector:(i,j)=("<<cur_coords[0]<<","<<cur_coords[1]<<"),the col rank of (0,,0) is "<<rank_col_root<<std::endl;

      	//std::cout<<"DEBUG:mpi_jacobi::distribute_vector:(i,j)=("<<cur_coords[0]<<","<<cur_coords[1]<<"),Allocation memory for count and displ array for scatterv="<<col_size<<std::endl;

      	int *disp = (int *)malloc(col_size*sizeof(int));
      	int *ncount = (int *)malloc(col_size*sizeof(int));

      	// for cart 0,0 processor whcih is sender
      	// prepare the count and disp arrays for scatterv
		if ( cur_col_rank == rank_col_root ){
			for (int i=0;i<col_size;i++) {
				ncount[i] = i < ( n % col_size ) ? cnp : fnp;
				disp[i] = i > 0 ? disp[i-1]+ncount[i-1] : 0;
          		//std::cout<<"DEBUG:mpi_jacobi::distribute_vector:(i,j)=("<<cur_coords[0]<<","<<cur_coords[1]<<"), allocated for col="<<i<<", disp="<<disp[i]<<",count="<<ncount[i]<<std::endl;
			}
		}

      	//std::cout<<"DEBUG:mpi_jacobi::distribute_vector:(i,j)=("<<cur_coords[0]<<","<<cur_coords[1]<<"),colrank="<<cur_col_rank<<",scatterv, with rank_root="<<rank_col_root<<std::endl;

      	// scatterv
		MPI_Scatterv(
			input_vector, ncount, disp, MPI_DOUBLE,
			tmp_vec,rcv_size, MPI_DOUBLE,
			rank_col_root, comm_col
		);

		//std::cout<<"DEBUG:mpi_jacobi::distribute_vector:(i,j)=("<<cur_coords[0]<<","<<cur_coords[1]<<"),Completed scatterv, with rank_root="<<rank_col_root<<std::endl;

		// print the rcv buffer
		//double *ptr = *local_vector;

		//for (int i=0;i<rcv_size;i++){
			//std::cout<<"DEBUG:mpi_jacobi::distribute_vector:(i,j)=("<<cur_coords[0]<<","<<cur_coords[1]<<"),recieved from scatterv local_vector["<<i<<"]="<<ptr[i]<<std::endl;
		//}

		free(ncount);
		free(disp);

    }


	//std::cout<<"DEBUG:mpi_jacobi::distribute_vector:(i,j)=("<<cur_coords[0]<<","<<cur_coords[1]<<"),completed"<<std::endl;

	// test
	//double *rr=nullptr;
	//double *row_vector=nullptr;
	//gather_vector(n,*local_vector,rr,comm);
	//transpose_bcast_vector(n,*local_vector,row_vector,comm);

}



// gather the local vector distributed among (i,0) to the processor (0,0)
void gather_vector(const int n, double* local_vector, double* output_vector, MPI_Comm comm)
{
	// TODO

	//std::cout<<"DEBUG:mpi_jacobi::gather_vector:begin with n="<<n<<std::endl;

	// rank in grid
	int cur_rank;
    MPI_Comm_rank(comm, &cur_rank);

  	// i,j coord in grid
	int cur_coords[2];
	MPI_Cart_coords(comm, cur_rank, 2, cur_coords);

  	// rank of root 0,0 matrix in grid
	int rank_root;
	int coords_root[2]={0,0};
	MPI_Cart_rank(comm, coords_root, &rank_root);

	// create a comm group for each column
	MPI_Comm comm_col;
	int cdims[2] = {1,0};
	MPI_Cart_sub(comm, cdims, &comm_col);

  	//std::cout<<"DEBUG:mpi_jacobi::gather_vector:(i,j)=("<<cur_coords[0]<<","<<cur_coords[1]<<"),created cart sub for rank="<<cur_rank<<std::endl;

  	// gather vector among first column processors only 
	if ( cur_coords[1]==0 ){

		//std::cout<<"DEBUG:mpi_jacobi::gather_vector:(i,j)=("<<cur_coords[0]<<","<<cur_coords[1]<<"),rank="<<cur_rank<<std::endl;
		// rank in the column
		int col_size;
		int cur_col_rank;
		MPI_Comm_size(comm_col,&col_size);
		MPI_Comm_rank(comm_col,&cur_col_rank);

		//std::cout<<"DEBUG:mpi_jacobi::gather_vector:(i,j)=("<<cur_coords[0]<<","<<cur_coords[1]<<"),colrank="<<cur_col_rank<<",col_size="<<col_size<<std::endl;

		// floor and ceil values
		int fnp = (int)floor(((double)n)/col_size);
		int cnp = (int)ceil( ((double)n)/col_size);

		// send buffer size at each processor
		int sendsize;
		if ( cur_col_rank < ( n % col_size ) ) 
			sendsize = cnp;
		else
			sendsize = fnp;


		// rank of root 0,0 matrix in the col comm to be used in scatterv
		int rank_col_root;
		MPI_Cart_rank(comm_col,coords_root,&rank_col_root);

		//std::cout<<"DEBUG:mpi_jacobi::gather_vector:(i,j)=("<<cur_coords[0]<<","<<cur_coords[1]<<"),the col rank of (0,,0) is "<<rank_col_root<<std::endl;

		//std::cout<<"DEBUG:mpi_jacobi::gather_vector:(i,j)=("<<cur_coords[0]<<","<<cur_coords[1]<<"),Allocation memory for count and displ array for scatterv="<<col_size<<std::endl;

		int *disp = (int *)malloc(col_size*sizeof(int));
		int *ncount = (int *)malloc(col_size*sizeof(int));

		// for cart 0,0 processor whcih is sender
		// prepare the count and disp arrays for scatterv
		if ( cur_col_rank == rank_col_root ){

			//std::cout<<"DEBUG:mpi_jacobi::gather_vector:(i,j)=("<<cur_coords[0]<<","<<cur_coords[1]<<"),Allocation memory for local vector n ="<<n<<std::endl;

			// memory is already allocated on rank 00 processor
			//output_vector=(double *)malloc(n*sizeof(double));
			for (int i=0;i<col_size;i++) {
				ncount[i] = i < ( n % col_size ) ? cnp : fnp;
				disp[i] = (i > 0) ? disp[i-1]+ncount[i-1] : 0;
				//std::cout<<"DEBUG:mpi_jacobi::gather_vector:(i,j)=("<<cur_coords[0]<<","<<cur_coords[1]<<"), allocated for col="<<i<<", disp="<<disp[i]<<",count="<<ncount[i]<<std::endl;
			}
		}

		//std::cout<<"DEBUG:mpi_jacobi::gather_vector:(i,j)=("<<cur_coords[0]<<","<<cur_coords[1]<<"),colrank="<<cur_col_rank<<",gatherv, with rank_root="<<rank_col_root<<std::endl;

		// scatterv
		MPI_Gatherv(
			local_vector, sendsize, MPI_DOUBLE, 
			output_vector, ncount,disp, MPI_DOUBLE,
			rank_col_root, comm_col
		);

		//std::cout<<"DEBUG:mpi_jacobi::gather_vector:(i,j)=("<<cur_coords[0]<<","<<cur_coords[1]<<"),Completed gatherv, with rank_root="<<rank_col_root<<std::endl;

		// print the rcv buffer
		//double *ptr = local_vector;

		//for (int i=0;i<sendsize;i++){
			//std::cout<<"DEBUG:mpi_jacobi::gather_vector:(i,j)=("<<cur_coords[0]<<","<<cur_coords[1]<<"),sent through gather local_vector["<<i<<"]="<<ptr[i]<<std::endl;
		//}

		//ptr = output_vector;

		//if ( cur_col_rank == rank_col_root ) {
		//  for (int i=0;i<n;i++){
		//std::cout<<"DEBUG:mpi_jacobi::gather_vector:(i,j)=("<<cur_coords[0]<<","<<cur_coords[1]<<"),recieved from gather output_vector["<<i<<"]="<<ptr[i]<<std::endl;
		//}
		//}

		free(ncount);
		free(disp);
	}

	//std::cout<<"DEBUG:mpi_jacobi::gather_vector:(i,j)=("<<cur_coords[0]<<","<<cur_coords[1]<<"),completed"<<std::endl;
}



void distribute_matrix(const int n, double* input_matrix, double** local_matrix, MPI_Comm comm)
{
	// n : number of processors
	// A : matrix for computation
	// x : for x in Ax = y


	int *disp = nullptr;
	int *ncount = nullptr;
	double *row_block_matrix = nullptr;
	int block_rows = 0;

	// rank in grid
	int cur_rank;
	int cur_coords[2];
	MPI_Comm_rank(comm, &cur_rank);

	// i,j coord in grid
	MPI_Cart_coords(comm, cur_rank, 2, cur_coords);


	// rank of root 0,0 matrix in grid
	int rank_root;
	int coords_root[2]={0, 0};

	MPI_Cart_rank(comm, coords_root, &rank_root);



	// 1.  Distribute the rows from processor 0.0 to all processors in the first column
	//
	// create a comm group for each column
	MPI_Comm comm_col;
	int cdims[2] = {1,0};
	MPI_Cart_sub(comm, cdims, &comm_col);
	// int MPI_Cart_sub(MPI_Comm comm, const int remain_dims[], MPI_Comm *newcomm)
	// Partitions a communicator into subgroups which form lower-dimensional cartesian subgrids

	// Input
	// comm : communicator with cartesian structure (handle)
	// remain_dims : the ith entry of remain_dims specifies whether the ith dimension is kept in the subgrid (true) or is dropped (false) (logical vector)
	// Output
	// newcomm : communicator containing the subgrid that includes the calling process (handle)




	//std::cout<<"DEBUG:mpi_jacobi::distribute_matrix:(i,j)=("<<cur_coords[0]<<","<<cur_coords[1]<<"),created column cart sub for rank="<<cur_rank<<std::endl;

	// rank in the column
	int col_size;
	int cur_col_rank;
	MPI_Comm_size(comm_col,&col_size);
	MPI_Comm_rank(comm_col,&cur_col_rank);

	// floor and ceil values
	int fnp = (int)floor(((double)n)/col_size);
	int cnp = (int)ceil( ((double)n)/col_size);

	// allocate memory for rcv buffer

	block_rows = cur_col_rank < ( n % col_size ) ? cnp : fnp;
	
	// first distribute among first column processors only 
	if ( cur_coords[1]==0 ){

		//std::cout<<"DEBUG:mpi_jacobi::distribute_matrix:(i,j)=("<<cur_coords[0]<<","<<cur_coords[1]<<"),rank="<<cur_rank<<",colrank="<<cur_col_rank<<",col_size="<<col_size<<std::endl;

		int rcv_size = block_rows*n;

		//std::cout<<"DEBUG:mpi_jacobi::distribute_matrix:(i,j)=("<<cur_coords[0]<<","<<cur_coords[1]<<"),Allocation memory for row_block_matrix="<<rcv_size<<std::endl;

		row_block_matrix=(double *)malloc(rcv_size*sizeof(double));

		// rank of root 0,0 matrix in the col comm to be used in scatterv
		int rank_col_root;
		int colcoords_root = 0;
		MPI_Cart_rank(comm_col,&colcoords_root,&rank_col_root);

		//std::cout<<"DEBUG:mpi_jacobi::distribute_matrix:(i,j)=("<<cur_coords[0]<<","<<cur_coords[1]<<"),the col rank of (0,0) is "<<rank_col_root<<std::endl;

		//std::cout<<"DEBUG:mpi_jacobi::distribute_matrix:(i,j)=("<<cur_coords[0]<<","<<cur_coords[1]<<"),Allocation memory for count and displ array for scatterv="<<col_size<<std::endl;

		//disp = (int *)malloc(col_size*sizeof(int));
		//ncount = (int *)malloc(col_size*sizeof(int));

		// for cart 0,0 processor whcih is sender
		// prepare the count and disp arrays for scatterv
		if ( cur_col_rank == rank_col_root ){

			disp = (int *)malloc(col_size*sizeof(int));
			ncount = (int *)malloc(col_size*sizeof(int));

			for (int i=0;i<col_size;i++) {
				ncount[i] = i < ( n % col_size ) ? n*cnp : n*fnp;
				disp[i] = i > 0 ? disp[i-1]+ncount[i-1] : 0;

				//std::cout<<"DEBUG:mpi_jacobi::distribute_matrix:n="<<n<<"(i,j)=("<<cur_coords[0]<<","<<cur_coords[1]<<"), allocated for col="<<i<<", disp="<<disp[i]<<",count="<<ncount[i]<<std::endl;

			}

		}

		//std::cout<<"DEBUG:mpi_jacobi::distribute_matrix:(i,j)=("<<cur_coords[0]<<","<<cur_coords[1]<<"),colrank="<<cur_col_rank<<",scatterv, with rank_root="<<rank_col_root<<std::endl;

		// scatterv
		MPI_Scatterv(
			input_matrix,ncount,disp, MPI_DOUBLE, 
			row_block_matrix,rcv_size, MPI_DOUBLE,
			rank_col_root, comm_col
		);

		//std::cout<<"DEBUG:mpi_jacobi::distribute_matrix:(i,j)=("<<cur_coords[0]<<","<<cur_coords[1]<<"),Completed scatterv, with rank_root="<<rank_col_root<<std::endl;

		// print the rcv buffer
		//double *ptr = row_block_matrix;

		//if ( cur_coords[0]==1) {
		//for (int i=0;i<rcv_size;i++){
		//std::cout<<"DEBUG:mpi_jacobi::distribute_matrix::n="<<n<<"(i,j)=("<<cur_coords[0]<<","<<cur_coords[1]<<"),recieved from scatterv row_block_matrix["<<i<<"]="<<ptr[i]<<std::endl;
		//}
		//}

		if ( cur_col_rank == rank_col_root ) {
			free(ncount);
			free(disp);
		}
	}

	//std::cout<<"DEBUG:mpi_jacobi::distribute_matrix:(i,j)=("<<cur_coords[0]<<","<<cur_coords[1]<<"),Completed distributing the rows in processor 0,0 to processors in first column "<<std::endl;




	// 2. distribute the columns from first processor in each row to al row processors
	//
	MPI_Barrier(comm);

	// 1.  Distribute the rows in the first column
	//
	// create a comm group for each row
	MPI_Comm comm_row;
	int rdims[2] = {0,1};
	MPI_Cart_sub(comm, rdims, &comm_row);

	//std::cout<<"DEBUG:mpi_jacobi::distribute_matrix:(i,j)=("<<cur_coords[0]<<","<<cur_coords[1]<<"),created row cart sub for rank="<<cur_rank<<std::endl;

	// rank in the row
	int row_size;
	int cur_row_rank;
	MPI_Comm_size(comm_row,&row_size);
	MPI_Comm_rank(comm_row,&cur_row_rank);

	//std::cout<<"DEBUG:mpi_jacobi::distribute_matrix:(i,j)=("<<cur_coords[0]<<","<<cur_coords[1]<<"),rowrank="<<cur_row_rank<<",row_size="<<row_size<<std::endl;

	// floor and ceil values
	fnp = (int)floor(((double)n)/row_size);
	cnp = (int)ceil( ((double)n)/row_size);

	int rcv_size;
	// allocate memory for rcv buffer
	rcv_size = cur_row_rank < ( n % row_size ) ? cnp*block_rows : fnp*block_rows;
	//std::cout<<"DEBUG:mpi_jacobi::distribute_matrix:(i,j)=("<<cur_coords[0]<<","<<cur_coords[1]<<"),allocating memory for local_matrix="<<rcv_size<<std::endl;


	double *temp_matrix;
	temp_matrix=(double *)malloc(rcv_size*sizeof(double));
	*local_matrix = &temp_matrix[0];

	// rank of root 0,0 matrix in the row comm to be used in scatterv
	int rank_row_root;
	int row_coords_root = 0;
	MPI_Cart_rank(comm_row, &row_coords_root, &rank_row_root);

	//std::cout<<"DEBUG:mpi_jacobi::distribute_matrix:(i,j)=("<<cur_coords[0]<<","<<cur_coords[1]<<"),the row rank of ("<<row_coords_root<<",0) is "<<rank_row_root<<std::endl;


	//disp = (int *)malloc(row_size*sizeof(int));
	//ncount = (int *)malloc(row_size*sizeof(int));

	MPI_Datatype  tmpcol,coltype;
	MPI_Type_vector(block_rows, 1, n, MPI_DOUBLE, &tmpcol);
	MPI_Type_create_resized(tmpcol, 0, 1*sizeof(double), &coltype);
	MPI_Type_commit(&coltype);
	MPI_Type_free(&tmpcol);

	// for cart 0,0 processor whcih is sender
	// prepare the count and disp arrays for scatterv
	if ( cur_row_rank == rank_row_root ){

		//std::cout<<"DEBUG:mpi_jacobi::distribute_matrix:(i,j)=("<<cur_coords[0]<<","<<cur_coords[1]<<"), allocating for row disp/count arrays "<<std::endl;
		disp = (int *)malloc(row_size*sizeof(int));
		ncount = (int *)malloc(row_size*sizeof(int));

		for (int i=0;i<row_size;i++) {
			ncount[i] = i < ( n % row_size ) ? cnp : fnp;
			disp[i] = (i>0) ? disp[i-1]+ncount[i-1] : 0;
			
			//std::cout<<"DEBUG:mpi_jacobi::distribute_matrix:(i,j)=("<<cur_coords[0]<<","<<cur_coords[1]<<"), allocated for col="<<i<<", disp="<<disp[i]<<",count="<<ncount[i]<<std::endl;

		}
	}

	//std::cout<<"DEBUG:mpi_jacobi::distribute_matrix:(i,j)=("<<cur_coords[0]<<","<<cur_coords[1]<<"),rowrank="<<cur_row_rank<<",scatterv, for row with rank_root="<<rank_row_root<<std::endl;

	MPI_Barrier(comm_row);

	// scatterv
	MPI_Scatterv(row_block_matrix,ncount,disp, coltype, temp_matrix,rcv_size, MPI_DOUBLE, rank_row_root,comm_row);


	//std::cout<<"DEBUG:mpi_jacobi::distribute_matrix:(i,j)=("<<cur_coords[0]<<","<<cur_coords[1]<<"),Completed scatterv, with rank_root="<<rank_row_root<<std::endl;

	// print the rcv buffer
	//double *ptr = *local_matrix;

	//if ( ( cur_coords[0] == 1 ) && (cur_coords[1] == 1) ){

	//for (int i=0;i<rcv_size;i++){
	//std::cout<<"DEBUG:mpi_jacobi::distribute_matrix::n="<<n<<"(i,j)=("<<cur_coords[0]<<","<<cur_coords[1]<<"),recieved from scatterv local_matrix["<<i<<"]="<<ptr[i]<<std::endl;
	//}
	//}

	MPI_Type_free(&coltype);

	if ( cur_row_rank == rank_row_root ){
		free(ncount);
		free(disp);
		free(row_block_matrix);
	}

	//std::cout<<"DEBUG:mpi_jacobi::distribute_matrix:(i,j)=("<<cur_coords[0]<<","<<cur_coords[1]<<"),Completed distributing the columns from column 0 "<<std::endl;

	//std::cout<<"DEBUG:mpi_jacobi::distribute_matrix:(i,j)=("<<cur_coords[0]<<","<<cur_coords[1]<<"),Completed distribute_matrix "<<std::endl;

}



void transpose_bcast_vector(const int n, double* col_vector, double* row_vector, MPI_Comm comm)
{
	// TODO
	//std::cout<<"DEBUG:mpi_jacobi::transpose_bcast_vector:begin with n="<<n<<std::endl;

	// rank in grid
	int cur_rank;
	int cur_coords[2];
	MPI_Comm_rank(comm, &cur_rank);
	// i,j coord in grid
	MPI_Cart_coords(comm, cur_rank, 2, cur_coords);


	//get the row and column block size for each processor
	MPI_Comm comm_row,comm_col;

	int rdims[2] = {0,1};
	MPI_Cart_sub(comm, rdims,&comm_row);
	int ldims[2] = {1,0};
	MPI_Cart_sub(comm,ldims,&comm_col);

	//std::cout<<"DEBUG:mpi_jacobi::transpose_bcast_vector:(i,j)=("<<cur_coords[0]<<","<<cur_coords[1]<<"),created row cart sub for rank="<<cur_rank<<std::endl;

	// rank in the row and col
	int row_size;
	int cur_row_rank;
	int col_size;
	int cur_col_rank;

	MPI_Comm_size(comm_row, &row_size);
	MPI_Comm_rank(comm_row, &cur_row_rank);

	MPI_Comm_size(comm_col, &col_size);
	MPI_Comm_rank(comm_col, &cur_col_rank);


	// rank of root 0,0 matrix in the row comm to be used in mpi_recv
	int rank_row_root;
	int row_coords_root = 0;
	MPI_Cart_rank(comm_row,&row_coords_root,&rank_row_root);

	// get the rank of the ii processor to send to
	int row_diag_proc_coords = cur_col_rank;
	int rank_row_diag;
	MPI_Cart_rank(comm_row, &row_diag_proc_coords, &rank_row_diag);

	// get the rank of the ii processor to send to
	int coldiagproccords = cur_row_rank;
	int rankcoldiag;
	MPI_Cart_rank(comm_col, &coldiagproccords, &rankcoldiag);


	//std::cout<<"DEBUG:mpi_jacobi::transpose_broadcast_vector:(i,j)=("<<cur_coords[0]<<","<<cur_coords[1]<<"),rowrank="<<cur_row_rank<<",row_size="<<row_size<<",colrank="<<cur_col_rank<<",col_size="<<col_size<<std::endl;

	// get the column block size
	int col_block_size;
	int fnp = (int)floor(((double)n)/col_size);
	int cnp = (int)ceil( ((double)n)/col_size);

	col_block_size = cur_col_rank < ( n % col_size ) ? cnp : fnp;

	//send from first col to i,i processors only
	if ( (cur_coords[1] == 0 ) ) {

		if (cur_coords[0] == 0 ) {
			for (int i=0;i<col_block_size;i++) {
				row_vector[i]=col_vector[i];
			}
		} else {
			MPI_Send(col_vector,col_block_size, MPI_DOUBLE,rank_row_diag,0,comm_row);
			//std::cout<<"DEBUG:mpi_jacobi::transpose_bcast_vector:(i,j)=("<<cur_coords[0]<<","<<cur_coords[1]<<"),sending local_vector to rank_row_diag="<<rank_row_diag<<std::endl;
		}
	}


	//rcv in i,i processors from 0 processor in same row
	if ( (cur_coords[0] == cur_coords[1]) && (cur_coords[1] != 0 ) ) {

		MPI_Recv(row_vector,col_block_size, MPI_DOUBLE,rank_row_root,0,comm_row,MPI_STATUS_IGNORE);
		//std::cout<<"DEBUG:mpi_jacobi::transpose_bcast_vector:(i,j)=("<<cur_coords[0]<<","<<cur_coords[1]<<"),recieving local_vector from rank_row_root="<<rank_row_root<<std::endl;
	}


	// Now broadcast from diagonal processor to all processors in a column
	MPI_Barrier(comm);


	fnp = (int)floor(((double)n)/row_size);
	cnp = (int)ceil( ((double)n)/row_size);
	int row_block_size = cur_row_rank < ( n % row_size ) ? cnp : fnp;

	//std::cout<<"DEBUG:mpi_jacobi::transpose_bcast_vector:(i,j)=("<<cur_coords[0]<<","<<cur_coords[1]<<"),broadcasting the row_vector of size="<<row_block_size<<",to all processors in a column, from processor col rank="<<rankcoldiag<<std::endl;

	// broadcast
	MPI_Bcast(row_vector, row_block_size, MPI_DOUBLE, rankcoldiag, comm_col);


	//std::cout<<"DEBUG:mpi_jacobi::transpose_bcast_vector:(i,j)=("<<cur_coords[0]<<","<<cur_coords[1]<<"),Completed transpose_bcast_vector "<<std::endl;

}


void distributed_matrix_vector_mult(const int n, double* local_A, double* local_x, double* local_y, MPI_Comm comm)
{
	// TODO

	//std::cout<<"DEBUG:mpi_jacobi::distributed_matrix_vector_mult:begin"<<std::endl;

	// rank in grid
	int cur_rank;
	int cur_coords[2];
	MPI_Comm_rank(comm, &cur_rank);

	// i,j coord in grid
	MPI_Cart_coords(comm, cur_rank, 2, cur_coords);

	//get the row and column block size for each processor
	MPI_Comm comm_row,comm_col;
	int rdims[2] = {0,1};
	MPI_Cart_sub(comm, rdims,&comm_row);
	int ldims[2] = {1,0};
	MPI_Cart_sub(comm,ldims,&comm_col);

	// rank in the row
	int row_size;
	int cur_row_rank;
	int col_size;
	int cur_col_rank;

	MPI_Comm_size(comm_row,&row_size);
	MPI_Comm_rank(comm_row,&cur_row_rank);
	MPI_Comm_size(comm_col,&col_size);
	MPI_Comm_rank(comm_col,&cur_col_rank);

	int rank_row_root;
	int row_coords_root = 0;
	MPI_Cart_rank(comm_row,&row_coords_root,&rank_row_root);

	//std::cout<<"DEBUG:mpi_jacobi::distributed_matrix_vector_mult:(i,j)=("<<cur_coords[0]<<","<<cur_coords[1]<<"),rowrank="<<cur_row_rank<<",row_size="<<row_size<<",colrank="<<cur_col_rank<<",col_size="<<col_size<<std::endl;

	// get the row block size
	int fnp = (int)floor(((double)n)/row_size);
	int cnp = (int)ceil( ((double)n)/row_size);
	int row_block_size = cur_row_rank < ( n % row_size ) ? cnp : fnp;

	// get the column block size
	fnp = (int)floor(((double)n)/col_size);
	cnp = (int)ceil(((double)n)/col_size);
	int col_block_size = cur_col_rank < ( n % col_size ) ? cnp : fnp;

	//std::cout<<"DEBUG:mpi_jacobi::distributed_matrix_vector_mult:(i,j)=("<<cur_coords[0]<<","<<cur_coords[1]<<"),allocating memory for x row_vector="<<row_block_size<<std::endl;

	double *row_vector=(double *)malloc(row_block_size*sizeof(double));

	transpose_bcast_vector(n,local_x,row_vector,comm);

	// multiply the n/p matrix in each processor
	for (int i = 0;i<col_block_size;i++){
		local_y[i]=0;
		for (int j=0;j<row_block_size;j++){
			local_y[i] = local_y[i] + local_A[col_block_size*j+i]*row_vector[j];  
		}
		//std::cout<<"DEBUG:mpi_jacobi::distributed_matrix_vector_mult:(i,j)=("<<cur_coords[0]<<","<<cur_coords[1]<<"),at local processor local_y["<<i<<"]="<<local_y[i]<<std::endl;
	}

	//sum up the results to the first column
	//std::cout<<"DEBUG:mpi_jacobi::distributed_matrix_vector_mult:(i,j)=("<<cur_coords[0]<<","<<cur_coords[1]<<"),completed local matrix multiplication, now reducing results to first column"<<std::endl;

	//reduce the matrix mult results to first column
	MPI_Barrier(comm);

	if ( cur_coords[1] == 0 ) 
		MPI_Reduce(MPI_IN_PLACE,local_y,col_block_size,MPI_DOUBLE,MPI_SUM,rank_row_root,comm_row);
	else
		MPI_Reduce(&local_y[0],nullptr,col_block_size,MPI_DOUBLE,MPI_SUM,rank_row_root,comm_row);

	free(row_vector);

	// print the results in the first col
	//if (cur_coords[1] == 0 ) {
	//for ( int i=0;i < col_block_size;i++) {
	//std::cout<<"DEBUG:mpi_jacobi::distributed_matrix_vector_mult:(i,j)=("<<cur_coords[0]<<","<<cur_coords[1]<<"),reduced y local_y["<<i<<"]="<<local_y[i]<<std::endl;
	//}
	//}

}


// Solves Ax = b using the iterative jacobi method
void distributed_jacobi(const int n, double* local_A, double* local_b, double* local_x,
			MPI_Comm comm, int max_iter, double l2_termination)
{

	// TODO
	//std::cout<<"DEBUG:mpi_jacobi::distributed_jacobi:begin"<<std::endl;


	// rank in grid
	int cur_rank;
	int cur_coords[2];
	MPI_Comm_rank(comm, &cur_rank);
	// i,j coord in grid
	MPI_Cart_coords(comm, cur_rank, 2, cur_coords);


	//get the row and column block size for each processor
	MPI_Comm comm_row,comm_col;
	int rdims[2] = {0,1};
	MPI_Cart_sub(comm, rdims,&comm_row);
	int ldims[2] = {1,0};
	MPI_Cart_sub(comm,ldims,&comm_col);

	//std::cout<<"DEBUG:mpi_jacobi::distributed_jacobi:(i,j)=("<<cur_coords[0]<<","<<cur_coords[1]<<"),created row cart sub for rank="<<cur_rank<<std::endl;

	// rank in the row and col
	int row_size;
	int cur_row_rank;
	int col_size;
	int cur_col_rank;

	MPI_Comm_size(comm_row,&row_size);
	MPI_Comm_rank(comm_row,&cur_row_rank);
	MPI_Comm_size(comm_col,&col_size);
	MPI_Comm_rank(comm_col,&cur_col_rank);


	// rank of root 0,0 matrix in the row comm to be used in mpi_recv
	int rank_row_root;
	int row_coords_root = 0;
	MPI_Cart_rank(comm_row,&row_coords_root,&rank_row_root);

	// get the rank of the ii processor to recv from
	int row_diag_proc_coords = cur_col_rank;
	int rank_row_diag;
	MPI_Cart_rank(comm_row,&row_diag_proc_coords,&rank_row_diag);

	//std::cout<<"DEBUG:mpi_jacobi::distributed_jacobi:(i,j)=("<<cur_coords[0]<<","<<cur_coords[1]<<"),rowrank="<<cur_row_rank<<",row_size="<<row_size<<",colrank="<<cur_col_rank<<",col_size="<<col_size<<std::endl;

	// get the row block size
	int row_block_size;
	int fnp = (int)floor(((double)n)/row_size);
	int cnp = (int)ceil( ((double)n)/row_size);
	row_block_size = cur_row_rank < ( n % row_size ) ? cnp : fnp;


	// get the column block size
	int col_block_size;
	fnp = (int)floor(((double)n)/col_size);
	cnp = (int)ceil( ((double)n)/col_size);
	col_block_size = cur_col_rank < ( n % col_size ) ? cnp : fnp;

	// remove local the non diagonal elements into r
	//double local_D[col_block_size];
	//double local_R[row_block_size*col_block_size];
	double *local_D = (double *) malloc(col_block_size*sizeof(double));

	for (int i = 0;i<col_block_size;i++){
		for (int j = 0;j<row_block_size;j++){
			////std::cout<<"A("<<i<<","<<j<<")="<<A[i*n+j];
			if ( (cur_coords[0] == cur_coords[1]) && ( i==j )){
				local_D[i]=local_A[j*col_block_size+i];
				//local_R[j*col_block_size+i] = 0.0;
				//std::cout<<"DEBUG:mpi_jacobi::distributed_jacobi:(i,j)=("<<cur_coords[0]<<","<<cur_coords[1]<<"),local_D["<<i<<"]="<<local_D[i]<<std::endl;
			}
			//else
			//	local_R[j*col_block_size+i]=local_A[j*col_block_size+i];

			//if (( cur_coords[1]==1) && (cur_coords[0] == 0 )) {
			//std::cout<<"DEBUG:mpi_jacobi::distributed_jacobi:(i,j)=("<<cur_coords[0]<<","<<cur_coords[1]<<"),local_R["<<i<<","<<j<<"]="<<local_R[j*col_block_size+i]<<std::endl;
			//}
		}
	}


	//send D to the first column
	//rcv in i,i processors from 0 processor in same row
	if ((cur_coords[0] == cur_coords[1]) && (cur_coords[1] != 0 ) ) {
		MPI_Send(local_D,col_block_size, MPI_DOUBLE,rank_row_root,0,comm_row);
		//std::cout<<"DEBUG:mpi_jacobi::distributed_jacobi:(i,j)=("<<cur_coords[0]<<","<<cur_coords[1]<<"),sending local_D from rank_row_diag="<<rank_row_diag<<", to rank_row_root"<<rank_row_root<<std::endl;
	}

	if ( (cur_coords[1] == 0 ) && (cur_coords[0] != 0) ) {
		MPI_Recv(local_D,col_block_size,MPI_DOUBLE,rank_row_diag,0,comm_row,MPI_STATUS_IGNORE);
		//std::cout<<"DEBUG:mpi_jacobi::distributed_jacobi:(i,j)=("<<cur_coords[0]<<","<<cur_coords[1]<<"),recieving local_D from rank_row_diag="<<rank_row_diag<<std::endl;
	}


	for (int i=0;i<col_block_size;i++){
		local_x[i] = 0; 
	}

	double l2norm = l2_termination+1;
	int iter = 0;

	//double local_y[col_block_size];
	double *local_y = (double *)malloc(col_block_size*sizeof(double));

	MPI_Barrier(comm);

	// y=Ax
	distributed_matrix_vector_mult(n,local_A,local_x,local_y,comm);

	while ( (iter < max_iter) && ( l2norm > l2_termination ) )
	{

		//std::cout<<"DEBUG:jacobi.cpp::distributed_jacobi:::(i,j)=("<<cur_coords[0]<<","<<cur_coords[1]<<"),starting Iteration="<<iter<<std::endl;


		//std::cout<<"DEBUG:jacobi.cpp::distributed_jacobi:::(i,j)=("<<cur_coords[0]<<","<<cur_coords[1]<<"),in Iteration="<<iter<<",crossed barrier"<<std::endl;

		// y = Ax
		///distributed_matrix_vector_mult(n,local_A,local_x,local_y,comm);

		//std::cout<<"DEBUG:jacobi.cpp::distributed_jacobi:::(i,j)=("<<cur_coords[0]<<","<<cur_coords[1]<<"),in Iteration="<<iter<<",completed Rx"<<std::endl;

		// first col operation
		//  get the new local_x
		if ( cur_coords[1] == 0 ) {
			// 1/D (b-Rx)
			for (int i = 0;i<col_block_size;i++){
				local_x[i]= (local_b[i]-(local_y[i]-(local_D[i]*local_x[i])))/local_D[i];
			}
		}

		//if ( cur_coords[1] == 0 ){
		//for (int i = 0;i<col_block_size;i++){
		//std::cout<<"DEBUG:mpi_jacobi::distributed_jacobi:(i,j)=("<<cur_coords[0]<<","<<cur_coords[1]<<"),multiplied y=rx, local_y["<<i<<"]="<<local_y[i]<<std::endl;
		//}
		//}

		// Ax
		distributed_matrix_vector_mult(n,local_A,local_x,local_y,comm);
		//std::cout<<"DEBUG:jacobi.cpp::distributed_jacobi:::(i,j)=("<<cur_coords[0]<<","<<cur_coords[1]<<"),in Iteration="<<iter<<",completed Ax"<<std::endl;

		//if ( cur_coords[1] == 0 ){
		//for (int i = 0;i<col_block_size;i++){
		//std::cout<<"DEBUG:mpi_jacobi::distributed_jacobi:(i,j)=("<<cur_coords[0]<<","<<cur_coords[1]<<"),multiplied y=ax local_y["<<i<<"]="<<local_y[i]<<std::endl;
		//}
		//}

		// ||Ax-B||
		double tl2norm=0;
		// first col operation
		if ( cur_coords[1] == 0 ) {
			for (int i = 0;i<col_block_size;i++){
				tl2norm = tl2norm + pow(local_b[i]-local_y[i],2);
			}

			//std::cout<<"DEBUG:jacobi.cpp::distributed_jacobi:::(i,j)=("<<cur_coords[0]<<","<<cur_coords[1]<<"),Iteration="<<iter<<", MPI_ALL_reduce sending tl2norm="<<tl2norm<<std::endl;

		}

		MPI_Barrier(comm);

		MPI_Allreduce(&tl2norm,&l2norm,1,MPI_DOUBLE,MPI_SUM,comm);

		l2norm = sqrt(l2norm);

		//std::cout<<"DEBUG:jacobi.cpp::distributed_jacobi:::(i,j)=("<<cur_coords[0]<<","<<cur_coords[1]<<"),completed Iteration="<<iter<<", with l2norm="<<l2norm<<std::endl;

		iter++;

	}

	free(local_D);
	free(local_y);

}

// wraps the distributed matrix vector multiplication
void mpi_matrix_vector_mult(const int n, double* A, double* x, double* y, MPI_Comm comm)
{
	// distribute the array onto local processors!
	double* local_A = nullptr;
	double* local_x = nullptr;
	distribute_matrix(n, &A[0], &local_A, comm);
	distribute_vector(n, &x[0], &local_x, comm);

	// allocate local result space
	double* local_y = new double[block_decompose_by_dim(n, comm, 0)];


	distributed_matrix_vector_mult(n, local_A, local_x, local_y, comm);

	// gather results back to rank 0
	gather_vector(n, local_y, y, comm);
}


// wraps the distributed jacobi function
void mpi_jacobi(const int n, double* A, double* b, double* x, MPI_Comm comm,
                int max_iter, double l2_termination)
{
	// distribute the array onto local processors!
	double* local_A = nullptr;
	double* local_b = nullptr;
	distribute_matrix(n, &A[0], &local_A, comm);
	distribute_vector(n, &b[0], &local_b, comm);

	// allocate local result space
	double* local_x = new double[block_decompose_by_dim(n, comm, 0)];
	distributed_jacobi(n, local_A, local_b, local_x, comm, max_iter, l2_termination);

	// gather results back to rank 0
	gather_vector(n, local_x, x, comm);

	// rank in grid
	int cur_rank;
	int cur_coords[2];
	MPI_Comm_rank(comm, &cur_rank);
	// i,j coord in grid
	MPI_Cart_coords(comm, cur_rank, 2, cur_coords);

	//if (( cur_coords[1] == 0 ) && (cur_coords[0] == 0)){
	//  std::cout<<"DEBUG:mpi_jacobi::distributed_jacobi:(i,j)=("<<cur_coords[0]<<","<<cur_coords[1]<<"),Results x = [";
	//  for (int i = 0;i<n;i++){
	//   std::cout<<x[i]<<" ,";
	//}
	// std::cout<<" ]"<<std::endl;
	//}

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

	char sep ='\0';
	if(display){
		for(int i = 0 ; i < n_row; ++i){
			for(int j = 0 ; j < n_col; ++j){
				printf("%c%.2f", sep, A[i*n_col+j]);
				sep = ',';
			}
			printf("\n");
			sep ='\0';
		}
	}
}

void gen_b(const int grid_size, double* b){
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
}


int main(int argc, char** argv){

	const int grid_size = 4;	// generate 4 x 4 grid

	double* A = new double[grid_size*grid_size*grid_size*grid_size];
	gen_matrix(grid_size, A);



    // MPI_Init(&argc, &argv);
    // int n = 4;

    // double A[4*4] = {
    //                     10., -1., 2., 0.,
    //                     -1., 11., -1., 3.,
    //                     2., -1., 10., -1.,
    //                     0.0, 3., -1., 8.
    //                 };

    // double x[4] =  {6., 25., -11., 15.};
    // double y[4];
    // double expected_y[4] = {13.,  325., -138.,  206.};

    // MPI_Comm grid_comm;
    // get_grid_comm(&grid_comm);

    // mpi_matrix_vector_mult(n, A, x, y, grid_comm);

    // for(int i = 0 ; i < n; ++i){
    //     printf("%.2f ", y[i]);
    // }
    // printf("\n");

    // MPI_Finalize();

    return 0;
}