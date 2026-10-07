#include "gf256_internal.h"

#include <stdlib.h>

/*
 * Minimum job size for using the worker pool.
 *
 * For smaller jobs, the overhead of waking worker threads and synchronizing
 * their completion can cost more than the parallel execution saves.
 */
#define PAR_THRESHOLD 32768u


/*
 *
 * The worker pool uses one mutex and two condition variables:
 *
 *   mutex
 *       Protects all shared worker-pool state.
 *
 *   wake
 *       Wakes workers when a new job is available or when the pool is
 *       shutting down.
 *
 *   done
 *       Wakes the calling thread after all worker threads have completed
 *       the current job.
 *
 * `active` is a monotonically increasing job generation number. Each worker
 * keeps its own `seen` generation and only processes a job when:
 *
 *       seen != active
 *
 * This makes spurious condition-variable wakeups harmless and prevents a
 * worker from processing the same job more than once.
 *
 * The current job is described by:
 *
 *   job       - function to execute
 *   job_arg   - argument passed to that function
 *   job_n     - total number of elements in the job
 *
 * Workers copy these values while holding the mutex and release the mutex
 * before executing the job. This is important: the actual computation must
 * not be serialized by the worker-pool mutex.
 *
 * The calling thread handles worker ID 0's range itself. Therefore,
 * `completed` counts only worker threads, and gf256_parallel() waits until
 * `threads - 1` workers have finished.
 *
 * All reads and writes of `stop`, `active`, `job`, `job_arg`, and
 * `completed` must happen while holding `mutex`.
 */


/*
 * Entry point for each worker thread.
 *
 * Each worker waits for a new job, calculates the portion of the input
 * assigned to its worker ID, executes that portion without holding the
 * pool mutex, and then reports completion.
 */
static void *worker_main(void *arg)
{
    gf256_worker *worker = arg;
    gf256_ctx *ctx = worker->ctx;

    /*
     * `seen_job` identifies the most recent job processed by this worker.
     * It is intentionally local to the worker thread.
     */
    unsigned seen_job = 0;

    pthread_mutex_lock(&ctx->mutex);

    for (;;) {
        /*
         * Wait until either:
         *
         *   - the pool is shutting down, or
         *   - a new job has been posted.
         *
         * pthread_cond_wait() may wake spuriously, so the condition must
         * always be checked again after returning.
         */
        while (!ctx->stop && seen_job == ctx->active) {
            pthread_cond_wait(&ctx->wake, &ctx->mutex);
        }

        if (ctx->stop) {
            break;
        }

        /*
         * Mark this job as seen before releasing the mutex. This guarantees
         * that this worker cannot accidentally process the same job twice.
         */
        seen_job = ctx->active;

        /*
         * Copy the job information while holding the mutex. The mutex can
         * then be released while the potentially expensive computation runs.
         */
        gf256_job_fn job = ctx->job;
        void *job_arg = ctx->job_arg;
        size_t element_count = ctx->job_n;

        /*
         * Divide the element range evenly among all participating threads.
         *
         * Worker 0 is handled by the calling thread, so worker IDs start at
         * 1 here.
         */
        size_t start = element_count * worker->id / ctx->threads;
        size_t end = element_count * (worker->id + 1) / ctx->threads;

        pthread_mutex_unlock(&ctx->mutex);

        /* Execute this worker's portion without holding the pool mutex. */
        job(job_arg, start, end);

        pthread_mutex_lock(&ctx->mutex);

        /*
         * Notify the calling thread when the last worker finishes.
         *
         * The calling thread handles worker 0's range itself, so only
         * `threads - 1` worker completions are required.
         */
        ++ctx->completed;
        if (ctx->completed == ctx->threads - 1) {
            pthread_cond_signal(&ctx->done);
        }
    }

    pthread_mutex_unlock(&ctx->mutex);

    return NULL;
}


/*
 * Execute a job using the worker pool.
 *
 * For small jobs, or when only one thread is configured, execute the job
 * directly on the calling thread. Otherwise, publish the job to the worker
 * pool and process worker 0's range on the calling thread while the other
 * workers process their ranges concurrently.
 */
void gf256_parallel(gf256_ctx *ctx,
                    size_t element_count,
                    gf256_job_fn job,
                    void *job_arg)
{
    /*
     * Avoid thread-pool overhead for single-threaded execution and small
     * operations.
     */
    if (ctx->threads == 1 || element_count < PAR_THRESHOLD) {
        job(job_arg, 0, element_count);
        return;
    }

    /*
     * Publish the new job while holding the mutex so that no worker can
     * observe partially updated job state.
     */
    pthread_mutex_lock(&ctx->mutex);

    ctx->job = job;
    ctx->job_arg = job_arg;
    ctx->job_n = element_count;
    ctx->completed = 0;

    /*
     * Increment the job generation before waking workers. Every worker that
     * observes the new generation will process this job exactly once.
     */
    ++ctx->active;

    pthread_cond_broadcast(&ctx->wake);
    pthread_mutex_unlock(&ctx->mutex);

    /*
     * The calling thread acts as worker 0. This avoids creating an extra
     * thread and ensures all configured threads can contribute to the job.
     */
    job(job_arg, 0, element_count / ctx->threads);

    /*
     * Wait until every worker thread has completed its portion.
     *
     * Use a while loop because condition-variable wakeups can be spurious.
     */
    pthread_mutex_lock(&ctx->mutex);

    while (ctx->completed < ctx->threads - 1) {
        pthread_cond_wait(&ctx->done, &ctx->mutex);
    }

    pthread_mutex_unlock(&ctx->mutex);
}


/*
 * Initialize the worker pool for a GF(256) context.
 *
 * Returns non-zero on success and zero if any synchronization primitive or
 * worker thread cannot be initialized.
 *
 * If creating a worker fails partway through initialization, all workers
 * that were already created are stopped and joined before the partially
 * initialized pool is destroyed.
 */
int gf256_threads_init(gf256_ctx *ctx)
{
    /*
     * No worker pool is required when operating with a single thread.
     */
    if (ctx->threads == 1) {
        return 1;
    }

    if (pthread_mutex_init(&ctx->mutex, NULL) != 0) {
        return 0;
    }

    if (pthread_cond_init(&ctx->wake, NULL) != 0) {
        pthread_mutex_destroy(&ctx->mutex);
        return 0;
    }

    if (pthread_cond_init(&ctx->done, NULL) != 0) {
        pthread_cond_destroy(&ctx->wake);
        pthread_mutex_destroy(&ctx->mutex);
        return 0;
    }

    /*
     * Worker 0 is the calling thread, so only threads - 1 worker structures
     * are needed.
     */
    ctx->workers = calloc(ctx->threads - 1, sizeof(*ctx->workers));

    if (ctx->workers == NULL) {
        pthread_cond_destroy(&ctx->done);
        pthread_cond_destroy(&ctx->wake);
        pthread_mutex_destroy(&ctx->mutex);
        return 0;
    }

    /*
     * Worker IDs start at 1 because worker 0 is always the calling thread.
     */
    for (unsigned worker_id = 1; worker_id < ctx->threads; ++worker_id) {
        gf256_worker *worker = &ctx->workers[worker_id - 1];

        worker->ctx = ctx;
        worker->id = worker_id;

        if (pthread_create(&worker->thread,
                           NULL,
                           worker_main,
                           worker) != 0) {
            /*
             * `stop` must be modified while holding the mutex because
             * worker_main() reads it while holding the same mutex.
             *
             * This is important even during initialization: some workers
             * may already be running when creation of a later worker fails.
             */
            pthread_mutex_lock(&ctx->mutex);

            ctx->stop = 1;
            pthread_cond_broadcast(&ctx->wake);

            pthread_mutex_unlock(&ctx->mutex);

            /*
             * Join every worker that was successfully created before the
             * failure. `worker_id` is the ID of the worker whose creation
             * just failed, so workers 1 .. worker_id - 1 are valid.
             */
            while (--worker_id > 0) {
                pthread_join(ctx->workers[worker_id - 1].thread, NULL);
            }

            free(ctx->workers);

            pthread_cond_destroy(&ctx->done);
            pthread_cond_destroy(&ctx->wake);
            pthread_mutex_destroy(&ctx->mutex);

            return 0;
        }
    }

    return 1;
}


/*
 * Shut down and destroy the worker pool.
 *
 * All workers are signaled to exit and joined before their resources and
 * synchronization primitives are destroyed.
 */
void gf256_threads_free(gf256_ctx *ctx)
{
    /*
     * There is no pool to destroy when only the calling thread is used.
     */
    if (ctx->threads == 1) {
        return;
    }

    /*
     * Set the shutdown flag while holding the mutex so that worker_main()
     * observes it safely.
     */
    pthread_mutex_lock(&ctx->mutex);

    ctx->stop = 1;
    pthread_cond_broadcast(&ctx->wake);

    pthread_mutex_unlock(&ctx->mutex);

    /*
     * Wait for every worker to exit before destroying shared state.
     */
    for (unsigned worker_id = 1;
         worker_id < ctx->threads;
         ++worker_id) {
        pthread_join(ctx->workers[worker_id - 1].thread, NULL);
    }

    free(ctx->workers);

    pthread_cond_destroy(&ctx->done);
    pthread_cond_destroy(&ctx->wake);
    pthread_mutex_destroy(&ctx->mutex);
}
