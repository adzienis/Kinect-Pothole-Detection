#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <assert.h>
#include <time.h>
#include <string.h>
#include <omp.h>

MPI_Comm COMM;

// Creates an array of random floats. Each number has a value from 0 - 1
float* create_rand_nums(const int num_elements) {
  float *rand_nums = (float *)malloc(sizeof(float) * num_elements);
  assert(rand_nums != NULL);
  for (int i = 0; i < num_elements; i++) {
    rand_nums[i] = (rand() / (float)RAND_MAX);
  }
  return rand_nums;
}

// Creates an array of random floats. Each number has a value from 0 - 1
float* create_ord_nums(const int num_elements) {
  float *ord_nums = (float *)malloc(sizeof(float) * num_elements);
  assert(ord_nums != NULL);
  for (int i = 0; i < num_elements; i++) {
    ord_nums[i] = i;//(float)i/(float)((num_elements*(num_elements-1))/2);
  }
  return ord_nums;
}

// Distance**2 between d-vectors pointed to by v1, v2.
static inline float distance2(const float *v1, const float *v2, const int d) {
  float dist = 0.0;
  for (int i=0; i<d; i++) {
    float diff = v1[i] - v2[i];
    dist += diff * diff;
  }
  return dist;
}

// Assign a site to the correct cluster by computing its distances to
// each cluster centroid.
int assign_site(const float* site, float* centroids,
		const int k, const int d) {
  int best_cluster = 0;
  float best_dist = distance2(site, centroids, d);
  float best_distances[100];
  
  for (int c = 1; c < k; c++) {
    float* centroid = centroids + c*d;
    float dist = distance2(site, centroid, d);
    if (dist < best_dist) {
      {
      best_cluster = c;
      best_dist = dist;
      }
    }
  }

  return best_cluster;
}


// Add a site (vector) into a sum of sites (vector).
static inline void add_site(const float * site, float * sum, const int d) {
  for (int i=0; i<d; i++) {
    sum[i] += site[i];
  }
}

// Print the centroids one per line.
void print_centroids(float * centroids, const int k, const int d) {
  float *p = centroids;
  printf("Centroids:\n");
  for (int i = 0; i<k; i++) {
    for (int j = 0; j<d; j++, p++) {
      printf("%f ", *p);
    }
    printf("\n");
  }
}

void kmeans(float* all_sites, int data_length, int k, int d, int rank, int nprocs, int* all_labels, float* centroids) {
  
  // Get stuff from command line:
    // number of sites per processor.
    // number of processors comes from mpirun command line.  -n

  int sites_per_proc = data_length/nprocs;
  int sites_at_root = data_length % nprocs + sites_per_proc;
  int* sites_per_proc_array = calloc(sizeof(int), nprocs);
  int* displ_per_proc_array = calloc(sizeof(int), nprocs);

  int* labels_per_proc_array = calloc(sizeof(int), nprocs);
  int* labels_displ_per_proc_array = calloc(sizeof(int), nprocs);

  int test_accum = 0;
  for(int i = 0; i < nprocs; i++) {
      test_accum += i == 0 ? sites_at_root : sites_per_proc;
      labels_per_proc_array[i] = i == 0 ? sites_at_root : sites_per_proc;
      labels_displ_per_proc_array[i] = sites_at_root*(i > 0) + sites_per_proc*(i-1)*(i > 1);

      sites_per_proc_array[i] = i == 0 ? sites_at_root*d : sites_per_proc*d;
      displ_per_proc_array[i] = sites_at_root*d*(i > 0) + sites_per_proc*d*(i-1)*(i > 1);
  }
  
  if(rank == 0) 
    sites_per_proc = sites_at_root;
  
  // Seed the random number generator to get different results each time
  //  srand(time(NULL));
  // No, we'd like the same results.
  srand(31359);


  //
  // Data structures in all processes.
  //
  // The sites assigned to this process.
  float* sites;  
  assert(sites = malloc(sites_per_proc * d * sizeof(float)));
  // The sum of sites assigned to each cluster by this process.
  // k vectors of d elements.
  float* sums;
  assert(sums = malloc(k * d * sizeof(float)));
  // The number of sites assigned to each cluster by this process. k integers.
  int* counts;
  assert(counts = malloc(k * sizeof(int)));
  // The cluster assignments for each site.
  int* labels;
  assert(labels = malloc(sites_per_proc * sizeof(int)));
  
  //
  // Data structures maintained only in root process.
  //
  // Sum of sites assigned to each cluster by all processes.
  float* grand_sums = NULL;
  // Number of sites assigned to each cluster by all processes.
  int* grand_counts = NULL;
  if (rank == 0) {
    
    // Take the first k sites as the initial cluster centroids.
    for (int i = 0; i < k * d; i++) {
      centroids[i] = all_sites[i]; 
    }
    //print_centroids(centroids, k, d);
    assert(grand_sums = malloc(k * d * sizeof(float)));
    assert(grand_counts = malloc(k * sizeof(int)));
  }

  // Root sends each process its share of sites.
  //MPI_Scatter(all_sites,d*sites_per_proc, MPI_FLOAT, sites,
  //            d*sites_per_proc, MPI_FLOAT, 0, COMM);
  MPI_Scatterv(all_sites,sites_per_proc_array, displ_per_proc_array, MPI_FLOAT, sites,
              d*sites_per_proc, MPI_FLOAT, 0, COMM);

  
  float norm = 1.0;  // Will tell us if centroids have moved.

  float accum = 0.0;

  while (norm > 0.0001) { // While they've moved...

    // Broadcast the current cluster centroids to all processes.
    MPI_Bcast(centroids, k*d, MPI_FLOAT,0, COMM);

    // Each process reinitializes its cluster accumulators.
    memset(sums, 0, k*d*sizeof(int));
    memset(counts, 0, k*sizeof(int));

    // Find the closest centroid to each site and assign to cluster.
    float* site = sites;
    for (int i = 0; i < sites_per_proc; i++) {
      int cluster = assign_site(site + i*d, centroids, k, d);
      // Record the assignment of the site to the cluster.
      counts[cluster]++;
      add_site(site + i*d, &sums[cluster*d], d);  
    }


    // Gather and sum at root all cluster sums for individual processes.
    MPI_Reduce(sums, grand_sums, k * d, MPI_FLOAT, MPI_SUM, 0, COMM);
    MPI_Reduce(counts, grand_counts, k, MPI_INT, MPI_SUM, 0, COMM);

    if (rank == 0) {
      // Root process computes new centroids by dividing sums per cluster
      // by count per cluster.
      for (int i = 0; i<k; i++) {
	for (int j = 0; j<d; j++) {
	  int dij = d*i + j;
	  grand_sums[dij] /= grand_counts[i];
	}
      }
      // Have the centroids changed much?
      norm = distance2(grand_sums, centroids, d*k);

      //printf("norm: %f\n",norm);
      // Copy new centroids from grand_sums into centroids.
      for (int i=0; i<k*d; i++) {
	centroids[i] = grand_sums[i];
      }
      //print_centroids(centroids,k,d);
    }
    // Broadcast the norm.  All processes will use this in the loop test.
    MPI_Bcast(&norm, 1, MPI_FLOAT, 0, COMM);
  }


  //if(rank == 0)
  //  printf("inner loop: %f\n", accum/CLOCKS_PER_SEC);

  // Now centroids are fixed, so compute a final label for each site.
  float* site = sites;
  for (int i = 0; i < sites_per_proc; i++, site += d) {

    labels[i] = assign_site(site, centroids, k, d);
  }

  /*if(rank == 0) {
    MPI_Send(centroids, k*d, MPI_FLOAT, 0, 6, MPI_COMM_WORLD);
  }*/

  // Gather all labels into root process.
//MPI_Gather(labels, sites_per_proc, MPI_INT, all_labels, sites_per_proc, MPI_INT, 0, MPI_COMM_WORLD);

  //  MPI_Gather(labels, sites_per_proc, MPI_INT,
  //      all_labels, sites_per_proc, MPI_INT, 0, COMM);

  MPI_Gatherv(labels, sites_per_proc, MPI_INT,
	     all_labels, labels_per_proc_array, labels_displ_per_proc_array, MPI_INT, 0, COMM);
  

  /*if((rank == 0) && 1) {
    print_centroids(centroids, k, d);
  }*/

  // Root can print out all sites and labels.
  /*if ((rank == 0) && 1) {
    float* site = all_sites; 
    for (int i = 0;
	 i < nprocs * sites_per_proc;
	 i++, site += d) {
      printf("%4d\n", all_labels[i]);
    }
  }*/
      

}

int min(int a, int b) {
  return a < b ? a : b;
}

void read_in_chunks(float* data, int data_length)
{
  int chunk_size = 100;
  int offset = 0;

  for(int i = 0; i < data_length/chunk_size; i++)
    fread(data + i*chunk_size, sizeof(float), chunk_size, stdin);
  /*
  while(data_length > 0) {
      int read = min(chunk_size, data_length);
      fread(data + offset, sizeof(int), read, stdin);

      offset += read;
      data_length -= read;
  }*/
}

int main(int argc, char** argv) {
  // Initial MPI and find process rank and number of processes.
  MPI_Init(NULL, NULL);
  int world_rank, world_nprocs;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_nprocs);

  int color = 1;
  while(1)
  {

  MPI_Comm_split(MPI_COMM_WORLD, color, world_rank, &COMM);

  int rank, nprocs;

  MPI_Comm_rank(COMM, &rank);
  MPI_Comm_size(COMM, &nprocs);

  int data_length;
  int k;  // number of clusters.
  int d; // dimension of data.
  float* all_sites; 
  float* test_sites;

  int* all_labels;
  if(rank == 0){
      MPI_Recv(&data_length, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Recv(&k, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Recv(&d, 1, MPI_INT, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      assert(all_sites = calloc(sizeof(float), data_length*d));
      MPI_Recv(all_sites, data_length*d, MPI_FLOAT, 0, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  }

  MPI_Bcast(&data_length, 1, MPI_INT, 0, COMM);
  MPI_Bcast(&k, 1, MPI_INT, 0, COMM);
  MPI_Bcast(&d, 1, MPI_INT, 0, COMM);

  //all_sites = create_ord_nums(d * data_length);
  
  float* centroids;
  assert(centroids = malloc(k * d * sizeof(float)));
  assert(all_labels = malloc(data_length * sizeof(int)));

  kmeans(all_sites, data_length, k, d, rank, nprocs, all_labels, centroids);

  if(rank ==0 && 1){
    MPI_Send(centroids, k*d, MPI_FLOAT, 0, 6, MPI_COMM_WORLD);
    MPI_Send(all_labels, data_length, MPI_INT, 0, 7, MPI_COMM_WORLD);
  }

//free(centroids);
//  free(all_labels);

}

  MPI_Comm_free(&COMM);

  MPI_Finalize();

  return 0;
}
