//a simple OpenCL kernel which copies all pixels from A to B
kernel void identity(global const uchar* A, global uchar* B) {
	int id = get_global_id(0);
	B[id] = A[id];
}

kernel void hist_simple(global const uchar* A, global int* H /*, const int nr_bins */) { 
	int id = get_global_id(0);

	//assumes that H has been initialised to 0
	int bin_index = A[id];//take value as a bin index
	INFINITY;
	/* if (bin_index > nr_bins - 1)
	{
		bin_index = nr_bins - 1;	
	}
	*/
	atomic_inc(&H[bin_index]);
}

kernel void hist_cum(global int* H, global int* CH) {
	int id = get_global_id(0);
	int N = get_global_size(0);
	for (int i = id + 1; i < N; i++)
		atomic_add(&CH[i], H[id]);
    }

//taken from tutorial 3
//Hillis-Steele basic inclusive scan
//requires additional buffer B to avoid data overwrite 
kernel void scan_hs(global int* H, global int* CH) {
	int id = get_global_id(0);
	int N = get_global_size(0);
	global int* C;

	for (int stride = 1; stride < N; stride *= 2) {
		CH[id] = H[id];
		if (id >= stride)
			CH[id] += H[id - stride];

		barrier(CLK_GLOBAL_MEM_FENCE); //sync the step

		C = H; H = CH; CH = C; //swap A & B between steps
	}
}

//taken from tutorial 3
//Blelloch basic exclusive scan
kernel void scan_bl(global int* A) {
	int id = get_global_id(0);
	int N = get_global_size(0);
	int t;

	//up-sweep
	for (int stride = 1; stride < N; stride *= 2) {
		if (((id + 1) % (stride*2)) == 0)
			A[id] += A[id - stride];

		barrier(CLK_GLOBAL_MEM_FENCE); //sync the step
	}

	//down-sweep
	if (id == 0)
		A[N-1] = 0;//exclusive scan

	barrier(CLK_GLOBAL_MEM_FENCE); //sync the step

	for (int stride = N/2; stride > 0; stride /= 2) {
		if (((id + 1) % (stride*2)) == 0) {
			t = A[id];
			A[id] += A[id - stride]; //reduce 
			A[id - stride] = t;		 //move
		}

		barrier(CLK_GLOBAL_MEM_FENCE); //sync the step
	}
}

kernel void normalise(global int* CH, global int* norm) {
	int id = get_global_id(0);
	//CH[255] is the total number of pixels 
	norm[id] = CH[id] * (double)255 / CH[255];
}

kernel void back_proj(global uchar* A, global int* norm, global uchar* B) {
	int id = get_global_id(0);
	B[id] = norm[A[id]];
}

