// noabort.c — Override abort()/raise(SIGABRT) for driver crash survival
// Also intercepts pthread_create to install sigaltstack on ALL threads.
// Usage: LD_PRELOAD=/path/to/libnoabort.so ./app
#define _GNU_SOURCE
#include <stdio.h>
#include <signal.h>
#include <unistd.h>
#include <dlfcn.h>
#include <pthread.h>
#include <sys/mman.h>
#include <sched.h>
#include <stdlib.h>
#include <sys/syscall.h>

// Install alternate signal stack on current thread (8KB, separate from main stack)
static void install_sigaltstack(void) {
    stack_t ss;
    ss.ss_sp = mmap(NULL, 32768, PROT_READ | PROT_WRITE,
                    MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (ss.ss_sp != MAP_FAILED) {
        ss.ss_size = 32768;
        ss.ss_flags = 0;
        sigaltstack(&ss, NULL);
    }
}

// Wrapper for every new thread
typedef struct {
    void *(*real_start)(void*);
    void *real_arg;
} thread_wrapper_t;

static void *thread_wrapper_fn(void *arg) {
    thread_wrapper_t info = *(thread_wrapper_t*)arg;
    free(arg);
    install_sigaltstack();
    return info.real_start(info.real_arg);
}

// Intercept pthread_create to wrap every thread with sigaltstack
int pthread_create(pthread_t *thread, const pthread_attr_t *attr,
                   void *(*start_routine)(void*), void *arg) {
    static int (*real_pthread_create)(pthread_t*, const pthread_attr_t*,
                                      void*(*)(void*), void*) = NULL;
    if (!real_pthread_create) {
        real_pthread_create = dlsym(RTLD_NEXT, "pthread_create");
    }
    thread_wrapper_t *info = malloc(sizeof(thread_wrapper_t));
    if (!info) return real_pthread_create(thread, attr, start_routine, arg);
    info->real_start = start_routine;
    info->real_arg = arg;
    return real_pthread_create(thread, attr, thread_wrapper_fn, info);
}

// Install sigaltstack on the main thread too (constructor runs at load time)
__attribute__((constructor))
static void init_main_thread(void) {
    install_sigaltstack();
}

void abort(void) {
    static __thread int entered = 0;
    if (!entered) {
        entered = 1;
        static int abortCount = 0;
        int c = __sync_add_and_fetch(&abortCount, 1);
        if (c <= 20)
            fprintf(stderr, "[noabort] abort() #%d intercepted — thread %ld\n", c, (long)syscall(186));
    }
    // Non-main threads: exit cleanly to release locks
    if (syscall(186) != getpid()) {
        pthread_exit(NULL);
    }
    // Main thread: yield instead of hard spin (single-CPU safe)
    for (;;) sched_yield();
    __builtin_unreachable();
}

int raise(int sig) {
    if (sig == 6 /*SIGABRT*/) {
        static int rcount = 0;
        int c = __sync_add_and_fetch(&rcount, 1);
        if (c <= 20)
            fprintf(stderr, "[noabort] raise(SIGABRT) #%d — returning 0 (suppressed)\n", c);
        return 0;
    }
    // Use real raise for non-SIGABRT
    static int (*real_raise)(int) = NULL;
    if (!real_raise) real_raise = dlsym(RTLD_NEXT, "raise");
    return real_raise ? real_raise(sig) : kill(getpid(), sig);
}

void __assert_fail(const char *e, const char *f, unsigned int l, const char *fn) {
    static int c = 0;
    int n = __sync_add_and_fetch(&c, 1);
    if (n <= 20)
        fprintf(stderr, "[noabort] assert(%s) at %s:%u ignored\n", e, f, l);
}

void __fortify_fail(const char *msg) {
    static int c = 0;
    int n = __sync_add_and_fetch(&c, 1);
    if (n <= 20)
        fprintf(stderr, "[noabort] __fortify_fail(%s) ignored\n", msg);
}

void __stack_chk_fail(void) {
    static int c = 0;
    int n = __sync_add_and_fetch(&c, 1);
    if (n <= 20)
        fprintf(stderr, "[noabort] __stack_chk_fail ignored\n");
}
