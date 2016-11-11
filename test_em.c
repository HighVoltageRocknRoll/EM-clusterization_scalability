/*
||==========================================================================||
                            by Sergey Lavrushkin
||==========================================================================||
*/

#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define M_PI 3.14159265358979323846

double random_float()
{
    return (double)(rand()) / (double)(RAND_MAX);
}

// Random float in interval (0, 1).
double random_float_interval()
{
    return ((double)(rand()) + 1.0) / ((double)(RAND_MAX) + 2.0);
}

double random_uniform()
{
    double u0 = random_float_interval();
    double u1 = random_float_interval();

    return sqrt(-2.0 * log(u0)) * cos(2 * M_PI * u1);
}

double* generate_data(int n, int q, int k)
{
    const double min_mean = -10.0;
    const double max_mean = 10.0;
    const double min_std = 0.1;
    const double max_std = 2.5;

    double* data = (double*)malloc(n * q * sizeof(double));
    double* means = (double*)malloc(q * k * sizeof(double));
    double* stds = (double*)malloc(q * k * sizeof(double));

    int i, j;

    // Set clusters parameters.
    for (i = 0; i < k; ++i)
    {
        for (j = 0; j < q; ++j)
        {
            means[i * q + j] = min_mean + random_float() * (max_mean - min_mean);
            stds[i * q + j] = min_std + random_float() * (max_std - min_std);
            //printf("mean -- %f; std -- %f\n", means[i * q + j], stds[i * q + j]);
        }
    }

    // Generate samples.
    //printf("Generated data:\n");
    for (i = 0; i < n; ++i)
    {
        int cluster = rand() % k;
        for (j = 0; j < q; ++j)
        {
            data[i * q + j] = stds[cluster * q + j] * random_uniform() + means[cluster * q + j];

            //printf("%f ", data[i * q + j]);
        }
        //printf(" -- cluster %d\n", cluster);
    }

    free(means);
    free(stds);

    return data;
}

void init_gmm_params(double* weights, double* means, double* covs, double* dets_sqrt, int q, int k)
{
    int i, j;
    for (i = 0; i < k; ++i)
    {
        weights[i] = 1.0 / (double)k;
        dets_sqrt[i] = 1.0;
        for (j = 0; j < q; ++j)
        {
            means[i * q + j] = 2.0 * random_float() - 1.0;
            covs[i * q + j] = 1.0;
        }
    }
}

int main(int argc, char** argv)
{
    int rank;

    if (argc < 7)
    {
        return -1;
    }

    int data_size = atoi(argv[1]);
    int initial_size = atoi(argv[2]);
    int tests_number = atoi(argv[3]);
    int q = atoi(argv[4]);
    int k = atoi(argv[5]);
    int n_iter = atoi(argv[6]);

    srand(0);

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // Get number of slaves.
    int slaves_num;
    MPI_Comm_size(MPI_COMM_WORLD, &slaves_num);
    slaves_num -= 1;
    MPI_Status status;
    
    // Clusters weights.
    double* weights = (double*)malloc(k * sizeof(double));
    // Clusters means.
    double* means = (double*)malloc(k * q * sizeof(double));
    // Covariance matrices diagonals.
    double* covs = (double*)malloc(k * q * sizeof(double));
    // Square roots of covariance matrices determinants.
    double* dets_sqrt = (double*)malloc(k * sizeof(double));

    double* cluster_fractions = (double*)malloc(k * sizeof(double));

    int i, j, l, n, iter;

    if (rank == 0)
    {
        // Master.

        // Generate Data.
        double* data = generate_data(data_size, q, k);

        for (n = initial_size; n <= data_size; n += (data_size - initial_size) / (tests_number - 1))
        {
            // Send some points to each slave.
            for (i = 1; i <= slaves_num; ++i)
            {
                int count_common = (n / slaves_num) * q;
                int count = count_common + (int)(i == slaves_num) * (n % slaves_num) * q;
                MPI_Send((void*)(data + (i - 1) * count_common), count, MPI_DOUBLE, i, i, MPI_COMM_WORLD);
            }

            init_gmm_params(weights, means, covs, dets_sqrt, q, k);
            MPI_Bcast((void*)weights, k, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Bcast((void*)means, k * q, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Bcast((void*)covs, k * q, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Bcast((void*)dets_sqrt, k, MPI_DOUBLE, 0, MPI_COMM_WORLD);

            double* recv_buf_k = (double*)malloc(k * sizeof(double));
            double* recv_buf_kq = (double*)malloc(k * q * sizeof(double));

            double start, overall_time = 0.0;
            int slave;
            for (iter = 0; iter < n_iter; ++iter)
            {
                // For timing.
                MPI_Barrier(MPI_COMM_WORLD);
                start = MPI_Wtime();
                // M-step.
                // Compute cluster fractions.
                for (j = 0; j < k; ++j)
                {
                    cluster_fractions[j] = 0.0;
                }

                for (slave = 1; slave <= slaves_num; ++slave)
                {
                    MPI_Recv(recv_buf_k, k, MPI_DOUBLE, slave, slave, MPI_COMM_WORLD, &status);
                    for (j = 0; j < k; ++j)
                    {
                        cluster_fractions[j] += recv_buf_k[j];
                    }
                }
                MPI_Bcast((void*)cluster_fractions, k, MPI_DOUBLE, 0, MPI_COMM_WORLD);


                // Compute means.
                for (j = 0; j < k; ++j)
                {
                    for (l = 0; l < q; ++l)
                    {
                        means[j * q + l] = 0.0;
                    }
                }

                for (slave = 1; slave <= slaves_num; ++slave)
                {
                    MPI_Recv(recv_buf_kq, k * q, MPI_DOUBLE, slave, slave, MPI_COMM_WORLD, &status);
                    for (j = 0; j < k; ++j)
                    {
                        for (l = 0; l < q; ++l)
                        {
                            means[j * q + l] += recv_buf_kq[j * q + l];
                        }
                    }
                }
                MPI_Bcast((void*)means, k * q, MPI_DOUBLE, 0, MPI_COMM_WORLD);

                //MPI_Barrier(MPI_COMM_WORLD);

                // Compute covariance matrices.
                for (j = 0; j < k; ++j)
                {
                    for (l = 0; l < q; ++l)
                    {
                        covs[j * q + l] = 0.0;
                    }
                }

                for (slave = 1; slave <= slaves_num; ++slave)
                {
                    MPI_Recv(recv_buf_kq, k * q, MPI_DOUBLE, slave, slave, MPI_COMM_WORLD, &status);
                    for (j = 0; j < k; ++j)
                    {
                        for (l = 0; l < q; ++l)
                        {
                            covs[j * q + l] += recv_buf_kq[j * q + l];
                        }
                    }
                }
                MPI_Bcast((void*)covs, k * q, MPI_DOUBLE, 0, MPI_COMM_WORLD);

                // For timing.
                MPI_Barrier(MPI_COMM_WORLD);

                overall_time += MPI_Wtime() - start;

                if (iter == n_iter - 1)
                {
                    /*
                    MPI_Recv(weights, k, MPI_DOUBLE, 1, 1, MPI_COMM_WORLD, &status);
                    MPI_Recv(means, k * q, MPI_DOUBLE, 1, 1, MPI_COMM_WORLD, &status);
                    MPI_Recv(covs, k * q, MPI_DOUBLE, 1, 1, MPI_COMM_WORLD, &status);

                    printf("RESULTS\n");
                    printf("Clusters weights:\n");
                    for (int j = 0; j < k; ++j)
                    {
                        printf("%f ", weights[j]);
                    }
                    printf("\n");

                    printf("Clusters means:\n");
                    for (int i = 0; i < k; ++i)
                    {
                        for (int j = 0; j < q; ++j)
                        {
                            printf("%f ", means[i * q + j]);
                        }
                        printf("\n");
                    }

                    printf("Clusters covariance matrices dioganals:\n");
                    for (int i = 0; i < k; ++i)
                    {
                        for (int j = 0; j < q; ++j)
                        {
                            printf("%f ", covs[i * q + j]);
                        }
                        printf("\n");
                    }*/

                    printf("Average iteration time for %d samples and %d slaves: %f\n", n, slaves_num, overall_time / n_iter);
                }
            }
            free(recv_buf_kq);
            free(recv_buf_k);
        }

        free(data);
    }
    else
    {
        // Slave.

        for (n = initial_size; n <= data_size; n += (data_size - initial_size) / (tests_number - 1))
        {
            // Get Samples.
            int n_samples = n / slaves_num + (int)(rank == slaves_num) * (n % slaves_num);
            int count = n_samples * q;
            double* data = (double*)malloc(count * sizeof(double));
            MPI_Recv(data, count, MPI_DOUBLE, 0, rank, MPI_COMM_WORLD, &status);

            MPI_Bcast((void*)weights, k, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Bcast((void*)means, k * q, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Bcast((void*)covs, k * q, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Bcast((void*)dets_sqrt, k, MPI_DOUBLE, 0, MPI_COMM_WORLD);

            double* probabilities = (double*)malloc(n_samples * k * sizeof(double));

            for (iter = 0; iter < n_iter; ++iter)
            {
                // For timing.
                MPI_Barrier(MPI_COMM_WORLD);
                // E-step.
                // Compute latent variables.
                for (i = 0; i < n_samples; ++i)
                {
                    double norm_coef = 0.0;
                    for (j = 0; j < k; ++j)
                    {
                        double dist = 0.0;
                        for (l = 0; l < q; ++l)
                        {
                            double diff = data[i * q + l] - means[j * q + l];
                            dist += diff * diff / covs[j * q + l];
                        }
                        probabilities[i * k + j] = weights[j] * exp(-dist / 2.0) / pow(2.0 * M_PI, (double)q / 2.0) / dets_sqrt[j];
                        norm_coef += probabilities[i * k + j];
                    }

                    for (j = 0; j < k; ++j)
                    {
                        probabilities[i * k + j] /= norm_coef;
                    }
                }

                // M-step.
                // Compute cluster fractions.
                for (j = 0; j < k; ++j)
                {
                    cluster_fractions[j] = 0;
                    for (i = 0; i < n_samples; ++i)
                    {
                        cluster_fractions[j] += probabilities[i * k + j];
                    }
                }

                MPI_Send((void*)cluster_fractions, k, MPI_DOUBLE, 0, rank, MPI_COMM_WORLD);

                MPI_Bcast((void*)cluster_fractions, k, MPI_DOUBLE, 0, MPI_COMM_WORLD);

                // Compute weights.
                for (j = 0; j < k; ++j)
                {
                    weights[j] = cluster_fractions[j] / n;
                }

                // Compute means.
                for (j = 0; j < k; ++j)
                {
                    for (l = 0; l < q; ++l)
                    {
                        means[j * q + l] = 0.0;
                        for (i = 0; i < n_samples; ++i)
                        {
                            means[j * q + l] += probabilities[i * k + j] * data[i * q + l];
                        }
                    }
                }

                MPI_Send((void*)means, k * q, MPI_DOUBLE, 0, rank, MPI_COMM_WORLD);

                MPI_Bcast((void*)means, k * q, MPI_DOUBLE, 0, MPI_COMM_WORLD);

                for (j = 0; j < k; ++j)
                {
                    for (l = 0; l < q; ++l)
                    {
                        means[j * q + l] /= cluster_fractions[j];
                    }
                }

                // Compute covariance matrices.
                for (j = 0; j < k; ++j)
                {
                    for (l = 0; l < q; ++l)
                    {
                        covs[j * q + l] = 0.0;
                        for (i = 0; i < n_samples; ++i)
                        {
                            double diff = data[i * q + l] - means[j * q + l];
                            covs[j * q + l] += probabilities[i * k + j] * diff * diff;
                        }
                    }
                }

                MPI_Send((void*)covs, k * q, MPI_DOUBLE, 0, rank, MPI_COMM_WORLD);

                MPI_Bcast((void*)covs, k * q, MPI_DOUBLE, 0, MPI_COMM_WORLD);

                for (j = 0; j < k; ++j)
                {
                    for (l = 0; l < q; ++l)
                    {
                        covs[j * q + l] /= cluster_fractions[j];
                    }
                }

                // Square roots of covariance matrices determinants.
                for (j = 0; j < k; ++j)
                {
                    dets_sqrt[j] = 1.0;
                    for (l = 0; l < q; ++l)
                    {
                        dets_sqrt[j] *= covs[j * q + l];
                    }
                    dets_sqrt[j] = sqrt(dets_sqrt[j]);
                }

                // For timing.
                MPI_Barrier(MPI_COMM_WORLD);

                /*
                if (iter == n_iter - 1 && rank == 1)
                {
                    MPI_Send(weights, k, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
                    MPI_Send(means, k * q, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
                    MPI_Send(covs, k * q, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
                }*/
            }

            free(probabilities);
            free(data);
        }
    }

    free(cluster_fractions);
    free(weights);
    free(means);
    free(covs);
    free(dets_sqrt);

    MPI_Finalize();

    return 0;
}
