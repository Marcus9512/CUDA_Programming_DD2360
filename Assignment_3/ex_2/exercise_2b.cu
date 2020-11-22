#include <stdio.h>
#include <random>
#include <time.h>

#define NUM_PARTICLES 100000000
#define NUM_ITERATIONS 100
#define BLOCK_SIZE 64  //Number of threads

#define RANDOM_C 1000
#define RANDOM_V 10

#define VELOCITY_DEC 0.0001

// A particle
typedef struct {
	float3 pos;
	float3 velocity;
}Particle;

//Update the velocity of a particle given an particle array and an index
__device__ void updateVelocity(Particle* par, int index) {
	par[index].pos.x -= VELOCITY_DEC;
	par[index].pos.y -= VELOCITY_DEC;
	par[index].pos.z -= VELOCITY_DEC;
}

//Update the position of a particle given an particle array and an index
__device__ void updatePos(Particle* par, int index) {
	//par[index].pos = par[index].pos + par[index].velocity;
	par[index].pos = make_float3(par[index].pos.x + par[index].velocity.x,
		par[index].pos.y + par[index].velocity.y, par[index].pos.z + par[index].velocity.z);
}

//Kernal function
__global__ void particleSim(Particle* par, int len, int iterations) {
	const int id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id >= len) return;

	for(int i = 0; i < iterations; i++){
		updateVelocity(par, id);
		updatePos(par, id);
	}	
}

// Calculate simulation on CPU
void particleCPU(Particle* par, int len) {
	for (int i = 0; i < len; i++) {
		//update velocity
		par[i].pos.x -= VELOCITY_DEC;
		par[i].pos.y -= VELOCITY_DEC;
		par[i].pos.z -= VELOCITY_DEC;

		//update position
		par[i].pos = make_float3(par[i].pos.x + par[i].velocity.x, 
			par[i].pos.y + par[i].velocity.y, par[i].pos.z + par[i].velocity.z);
	}
}

// Evaluate if the cpu and gpu solution are the same
bool equivalent(Particle* p_cpu, Particle* p_gpu, int len){
	float margin = 0.00001;
	for (int i = 0; i < len; i++) {
		//printf("X: %f %f, Y: %f %f Z: %f %f \n", p_gpu[i].pos.x, p_cpu[i].pos.x, p_gpu[i].pos.y,p_cpu[i].pos.y , p_gpu[i].pos.z, p_cpu[i].pos.z);
		//Check position
		if (fabs(p_gpu[i].pos.x - p_cpu[i].pos.x) > margin ||
			fabs(p_gpu[i].pos.y - p_cpu[i].pos.y) > margin ||
			fabs(p_gpu[i].pos.z - p_cpu[i].pos.z) > margin) {
			return false;
		}
		//Check velocity
		if (fabs(p_gpu[i].velocity.x - p_cpu[i].velocity.x) > margin ||
			fabs(p_gpu[i].velocity.y - p_cpu[i].velocity.y) > margin ||
			fabs(p_gpu[i].velocity.z - p_cpu[i].velocity.z) > margin) {
			return false;
		}
	}
	return true;
}

void runSimulation() {

	//To ensure number of blocks is rounded up 
	dim3 numberOfBlocks((NUM_PARTICLES + BLOCK_SIZE - 1) / BLOCK_SIZE);
	dim3 numberOfThreads(BLOCK_SIZE);

	Particle* particles;
	Particle* particles_parallel;

	cudaMallocManaged(&particles, NUM_PARTICLES * sizeof(Particle));
	cudaMallocManaged(&particles_parallel, NUM_PARTICLES * sizeof(Particle));

	//Fill random values particles
	srand((unsigned int)time(NULL));
	for (int i = 0; i < NUM_PARTICLES; i++) {
		particles[i].pos.x = ((float)rand() / (float)RAND_MAX) * RANDOM_C;
		particles[i].pos.y = ((float)rand() / (float)RAND_MAX) * RANDOM_C;
		particles[i].pos.z = ((float)rand() / (float)RAND_MAX) * RANDOM_C;
			
		particles[i].velocity.x = ((float)rand() / (float)RAND_MAX) * RANDOM_V;
		particles[i].velocity.y = ((float)rand() / (float)RAND_MAX) * RANDOM_V;
		particles[i].velocity.z = ((float)rand() / (float)RAND_MAX) * RANDOM_V;

		particles_parallel[i].pos.x = particles[i].pos.x;
		particles_parallel[i].pos.y = particles[i].pos.y;
		particles_parallel[i].pos.z = particles[i].pos.z;

		particles_parallel[i].velocity.x = particles[i].velocity.x;
		particles_parallel[i].velocity.y = particles[i].velocity.y;
		particles_parallel[i].velocity.z = particles[i].velocity.z;

	}	

	
	particleSim << <numberOfBlocks, numberOfThreads >> > (particles_parallel, NUM_PARTICLES, NUM_ITERATIONS);
	cudaDeviceSynchronize();

	//Run simulation on CPU
	clock_t start = clock();
	for (int i = 0; i < NUM_ITERATIONS; i++) {
		//printf("%d\n",i);
		particleCPU(particles, NUM_PARTICLES);
	}	
	double time = (double)(clock() - start) / CLOCKS_PER_SEC;

	printf("CPU done in %f seconds!\n", time);

	bool res = equivalent(particles, particles_parallel, NUM_PARTICLES);
	
	// Free memory
	cudaFree(particles_parallel);
	cudaFree(particles);

	if (res) {
		printf("Comparing the output for each implementation, Correct!\n");
	}
	else {
		printf("Comparing the output for each implementation, Wrong \n");
	}

}
int main() {
	runSimulation();
}