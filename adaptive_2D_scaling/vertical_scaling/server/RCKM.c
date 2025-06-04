#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <stdint.h>
#include <stdio.h>
#include <stdarg.h>

#define GPU_MAX_NUM 4
#define MAX_TASKS_PER_GPU 10
#define WINDOW_SIZE 100
#define PERIOD_HISTORY_SIZE 20
#define OVERFLOW_RATE_A 1.2
#define OVERFLOW_RATE_D 1.6
#define OVERFLOW_RATE_DLLM 1.3
#define OVERFLOW_RATE_U 4.0
#define INCREASE_FACTOR 1.3

typedef struct high_priority_task
{
    double rate_window[WINDOW_SIZE];
    double current_period;
    int current_zeros;
    double min_period;
    double max_period;
    int period_history[PERIOD_HISTORY_SIZE];
    int period_index;
    double average_period;
    int active;
    double is_llm;
    double requests_rate;
    double limits_rate;
} high_priority_task;

high_priority_task tasks[GPU_MAX_NUM][MAX_TASKS_PER_GPU];
pthread_mutex_t locks[GPU_MAX_NUM];
double global_limiter[GPU_MAX_NUM] = {};
double high_requests_sum[GPU_MAX_NUM] = {};
double low_requests_sum[GPU_MAX_NUM] = {};
double low_limits_sum[GPU_MAX_NUM] = {};
double low_kernel_record[GPU_MAX_NUM] = {};
double UPPER_LIMIT = 10000000;
double CONSERVATIVE_LIMIT = 1000000;
double CONTROL_FACTOR = 0.01;

void init_tasks()
{
    for (int i = 0; i < GPU_MAX_NUM; i++)
    {
        pthread_mutex_init(&locks[i], NULL);
        for (int j = 0; j < MAX_TASKS_PER_GPU; j++)
        {
            tasks[i][j].active = 0;
        }
    }
}

int allocate_task_slot(int device)
{
    pthread_mutex_lock(&locks[device]);
    for (int i = 0; i < MAX_TASKS_PER_GPU; i++)
    {
        if (!tasks[device][i].active)
        {
            tasks[device][i].active = 1;
            pthread_mutex_unlock(&locks[device]);
            return i;
        }
    }
    pthread_mutex_unlock(&locks[device]);
    return -1; // No slot available
}

void free_task_slot(int device, int task_id)
{
    pthread_mutex_lock(&locks[device]);
    high_priority_task *task = &tasks[device][task_id];

    memset(task->rate_window, 0, sizeof(task->rate_window));
    task->current_period = 0;
    task->current_zeros = 0;
    task->min_period = 1000;
    task->max_period = 0;
    memset(task->period_history, 0, sizeof(task->period_history));
    task->period_index = 0;
    task->average_period = 0;
    task->active = 0;
    task->is_llm = 0;
    task->requests_rate = 0;
    task->limits_rate = 0;

    pthread_mutex_unlock(&locks[device]);
}

double update_average_period(int device, int task_id)
{
    high_priority_task *task = &tasks[device][task_id];
    double sum = 0;
    int count = 0;
    double min_period = 1000;

    for (int i = 0; i < PERIOD_HISTORY_SIZE; i++)
    {
        if (task->period_history[i] > 0)
        {
            sum += task->period_history[i];
            min_period = (min_period < task->period_history[i]) ? min_period : task->period_history[i];
            count++;
        }
    }
    if (count > 0)
    {
        task->average_period = sum / count;
    }
    else
    {
        task->average_period = 0;
    }
    return (min_period > 1) ? min_period : 2;
}

int shift_window(int device, int task_id, double new_value)
{
    high_priority_task *task = &tasks[device][task_id];
    int last_value = task->rate_window[0] > 0;
    for (int i = WINDOW_SIZE - 1; i > 0; --i)
    {
        task->rate_window[i] = task->rate_window[i - 1];
    }
    task->rate_window[0] = new_value;

    if (new_value > 0)
    {
        if (!last_value)
        {
            task->current_zeros = 0;
        }
        task->current_period += 1;
    }
    else
    {
        if (last_value)
        {
            if (task->current_period > task->max_period)
            {
                task->max_period = task->current_period;
                if (task->min_period == 0 && task->current_period > 6)
                {
                    task->min_period = task->current_period;
                }
            }
            if (task->min_period > task->current_period && task->current_period != 0 && task->current_period > 6)
            {
                task->min_period = task->current_period;
            }
            // Store cycles in the historical array and update the average cycle.
            task->period_history[task->period_index] = task->current_period;
            task->period_index = (task->period_index + 1) % PERIOD_HISTORY_SIZE;
            double min_period = update_average_period(device, task_id); // Consider adjusting update_average_period to use task-specific data

            double overflow_rate = task->min_period != 0 ? task->current_period / task->min_period : 0;
            if (task->is_llm > 0 && task->min_period != 0)
                overflow_rate = task->current_period / min_period;
            if ((task->is_llm == 0. && overflow_rate >= OVERFLOW_RATE_D || (task->is_llm > 0 && overflow_rate >= OVERFLOW_RATE_DLLM)) && overflow_rate < OVERFLOW_RATE_U)
            { // state -> EMERGENCY
                double factor = (task->is_llm > 0) ? 0.1 * CONTROL_FACTOR : CONTROL_FACTOR;
                CONSERVATIVE_LIMIT = UPPER_LIMIT * low_requests_sum[device] * factor > (int)(CONSERVATIVE_LIMIT / overflow_rate) ? UPPER_LIMIT * low_requests_sum[device] * CONTROL_FACTOR : (int)(CONSERVATIVE_LIMIT / overflow_rate);
            }
            else if (overflow_rate < OVERFLOW_RATE_A && task->period_index % 5 == 0)
            { // state -> RECOVERY
                CONSERVATIVE_LIMIT = UPPER_LIMIT < (int)(INCREASE_FACTOR * CONSERVATIVE_LIMIT) ? UPPER_LIMIT : (int)(INCREASE_FACTOR * CONSERVATIVE_LIMIT);
            }
            // else: state -> CONTENTION keep same values with the last one.
            task->current_period = 0; // reset
        }
        task->current_zeros += 1;
    }
    return task->current_zeros;
}

static void LOGGER(int level, const char *msg, ...)
{
    va_list args;
    va_start(args, msg);
    vfprintf(stderr, msg, args);
    va_end(args);
}

ssize_t rio_readn(int fd, void *usrbuf, size_t n)
{
    size_t nleft = n;
    ssize_t nread;
    char *bufp = usrbuf;

    while (nleft > 0)
    {
        if ((nread = read(fd, bufp, nleft)) < 0)
        {
            if (errno == EINTR)
                continue;
            else
                return -1;
        }
        else if (nread == 0)
        {
            break;
        }
        nleft -= nread;
        bufp += nread;
    }
    return (n - nleft);
}

ssize_t rio_writen(int fd, const void *usrbuf, size_t n)
{
    size_t nleft = n;
    ssize_t nwritten;
    const char *bufp = usrbuf;

    while (nleft > 0)
    {
        if ((nwritten = write(fd, bufp, nleft)) <= 0)
        {
            if (errno == EINTR)
                continue;
            else
                return -1;
        }
        nleft -= nwritten;
        bufp += nwritten;
    }
    return n;
}

static int open_listenfd(int device)
{
    char SOCKET_PATH[108];
    const int LISTENQ = 8;
    struct sockaddr_un name;
    int ret;
    int listenfd;

    /* Create local socket. */

    listenfd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (listenfd == -1)
    {
        fprintf(stderr, "socket failed: %s\n", strerror(errno));
    }

    memset(&name, 0, sizeof(name));
    sprintf(SOCKET_PATH, "/etc/gsharing/rate_%d.sock", device);

    name.sun_family = AF_UNIX;
    strncpy(name.sun_path, SOCKET_PATH, sizeof(name.sun_path));

    ret = unlink(SOCKET_PATH);
    if (ret == -1)
    {
        if (!access(SOCKET_PATH, F_OK))
            fprintf(stderr, "unlink failed: %s\n", strerror(errno));
    }

    ret = bind(listenfd, (const struct sockaddr *)&name, sizeof(name));
    if (ret == -1)
    {
        fprintf(stderr, "bind failed: %s\n", strerror(errno));
    }

    ret = listen(listenfd, LISTENQ);
    if (ret == -1)
    {
        fprintf(stderr, "listen failed: %s\n", strerror(errno));
    }
    return listenfd;
}

typedef struct
{
    int connfd;
    int device;
} thread_arg;

static void *listener_thread(void *arg)
{
    thread_arg *targ = (thread_arg *)arg;
    int connfd = targ->connfd;
    int device = targ->device;

    double recv_rate = 1.;
    double requests_rate = 1.;
    double limits_rate = 1.;
    double is_llm = 0.;
    int task_id = -1;

    // Read is_llm param
    if (rio_readn(connfd, (void *)&is_llm, sizeof(double)) != sizeof(double))
    {
        close(connfd);
        free(targ);
        return NULL;
    }
    // Read requests_rate limits_rate
    if (rio_readn(connfd, (void *)&requests_rate, sizeof(double)) != sizeof(double) ||
        rio_readn(connfd, (void *)&limits_rate, sizeof(double)) != sizeof(double))
    {
        close(connfd);
        free(targ);
        return NULL;
    }

    if (requests_rate > 0)
    {
        high_requests_sum[device] += requests_rate;
        task_id = allocate_task_slot(device);
        if (task_id == -1)
        {
            close(connfd);
            free(targ);
            return NULL;
        }
        tasks[device][task_id].is_llm = is_llm;
        tasks[device][task_id].requests_rate = requests_rate;
        tasks[device][task_id].limits_rate = limits_rate;
    }
    else
    {
        low_requests_sum[device] += (-requests_rate);
        low_limits_sum[device] += (-limits_rate);
    }

    // clean up
    void clean_up()
    {
        if (task_id >= 0)
            free_task_slot(device, task_id);
        close(connfd);
        free(targ);
    }

    LOGGER(1, "Received requests_rate: %f, limits_rate: %f, high_requests_sum[device]: %f \n", requests_rate, limits_rate, high_requests_sum[device]);
    while (1)
    {
        double recv_counter = -1;
        ssize_t n = read(connfd, &recv_counter, sizeof(double));
        if (n != sizeof(double))
        {
            if (n > 0)
                LOGGER(1, "read error: received unexpected byte count\n");
            break;
        }
        recv_rate = recv_counter;

        if (recv_counter >= 0)
        { // high or SLO-sensitive priority task
            int idle_counts = shift_window(device, task_id, recv_rate);

            pthread_mutex_lock(&locks[device]); // lock
            // Firstly, check if all active tasks have idle_counts greater than threshold
            int all_idle = 1;
            int other_high_idle = 1;
            for (int j = 0; j < MAX_TASKS_PER_GPU; j++)
            {
                if (tasks[device][j].active)
                {
                    if (tasks[device][j].current_zeros <= 10)
                    {
                        other_high_idle = 0;
                    }
                    else if (tasks[device][j].current_zeros <= 120)
                    { // TODO fix me, automatic detect threshold
                        all_idle = 0;
                        break;
                    }
                }
            }

            if (idle_counts <= WINDOW_SIZE)
            {
                global_limiter[device] = CONSERVATIVE_LIMIT * low_requests_sum[device]; // keep conservative quota to avoid impacts of aysnchronized CUDA execution
            }
            else if (all_idle)
            {
                CONSERVATIVE_LIMIT = UPPER_LIMIT;
                global_limiter[device] = CONSERVATIVE_LIMIT * low_limits_sum[device]; // scale up for non SLO-sensitive tasks
            }
            pthread_mutex_unlock(&locks[device]); // unlock
            double token_for_high_priority_task = UPPER_LIMIT * limits_rate;
            if (low_kernel_record[device] > 1 || other_high_idle == 0)
            {
                token_for_high_priority_task = UPPER_LIMIT * requests_rate;
            }
            if (rio_writen(connfd, (void *)&token_for_high_priority_task, sizeof(double)) != sizeof(double))
            {
                LOGGER(4, "rio_writen of high priority error\n");
                break;
            }
            // monitor the kernel size counts, For trace analysis 
            // if (idle_counts<200 && device==0) {
            //     double high_kernels = recv_counter < rate_limit_for_high_priority_task ? recv_counter : rate_limit_for_high_priority_task;
            //     double low_kernels = low_kernel_record[device] < global_limiter[device] ? low_kernel_record[device] : global_limiter[device];
            //     fprintf(stderr, "HIGH-KERNEL-COUNTS: %f, LOW-KERNEL-COUNTS: %f\n", high_kernels, low_kernels);
            // }
        }
        else
        { // for low or non SLO-sensitive priority task
            low_kernel_record[device] = -recv_counter;
            double token_for_low_priority_task = global_limiter[device] * (-requests_rate) / low_requests_sum[device]; // if there exist multiple non SLO-instensive task instances.
            pthread_mutex_lock(&locks[device]);
            if (rio_writen(connfd, (void *)&token_for_low_priority_task, sizeof(double)) != sizeof(double))
            {
                LOGGER(4, "rio_writen for low priority error\n");
                break;
            }
            pthread_mutex_unlock(&locks[device]);
        }
    }

    atexit(clean_up);
    if (requests_rate > 0)
    { // high_priority SLO-sensitive
        high_requests_sum[device] -= requests_rate;
        CONSERVATIVE_LIMIT = (1 - high_requests_sum[device]) * UPPER_LIMIT;
        global_limiter[device] = CONSERVATIVE_LIMIT * low_limits_sum[device];
    }
    else
    { // low_priority non SLO-sensitive
        low_requests_sum[device] -= (-requests_rate);
        low_limits_sum[device] -= (-limits_rate);
    }
    return NULL;
}

static void *limit_manager(void *v_device)
{
    int device = (int)(uintptr_t)v_device;
    int listenfd = open_listenfd(device);
    if (listenfd < 0)
    {
        return NULL;
    }

    struct sockaddr_storage clientaddr;
    socklen_t clientlen = sizeof(struct sockaddr_storage);

    while (1)
    {
        int connfd = accept(listenfd, (struct sockaddr *)&clientaddr, &clientlen);
        if (connfd < 0)
        {
            LOGGER(1, "accept error: %s\n", strerror(errno));
            continue;
        }

        pthread_t tid;
        thread_arg *targ = malloc(sizeof(thread_arg));
        if (!targ)
        {
            close(connfd);
            continue;
        }
        targ->connfd = connfd;
        targ->device = device;
        pthread_create(&tid, NULL, listener_thread, targ);
        pthread_detach(tid);
    }
}

int main()
{
    init_tasks();
    pthread_t threads[GPU_MAX_NUM];
    for (int i = 0; i < GPU_MAX_NUM; i++)
    {
        global_limiter[i] = UPPER_LIMIT;
        pthread_create(&threads[i], NULL, limit_manager, (void *)(uintptr_t)i);
    }
    for (int i = 0; i < GPU_MAX_NUM; i++)
    {
        pthread_join(threads[i], NULL);
    }
    return 0;
}
