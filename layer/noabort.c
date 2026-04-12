// noabort.c — Override abort()/raise() for driver crash survival
// Usage: LD_PRELOAD=/path/to/libnoabort.so ./app
// The NVIDIA driver calls abort()/raise(SIGABRT) after detecting internal corruption.
// This intercepts it and suspends the offending thread instead of dying.
#include <stdio.h>
#include <signal.h>
#include <unistd.h>

void abort(void) {
    static __thread int entered = 0;
    if (!entered) {
        entered = 1;
        static int abortCount = 0;
        int c = __sync_add_and_fetch(&abortCount, 1);
        if (c <= 20)
            fprintf(stderr, "[noabort] abort() #%d intercepted — suspending thread\n", c);
    }
    for (;;) pause();
    __builtin_unreachable();
}

// Override raise() to catch raise(SIGABRT) from driver
static int (*real_raise)(int) = 0;
int raise(int sig) {
    if (sig == 6 /*SIGABRT*/) {
        static int rcount = 0;
        int c = __sync_add_and_fetch(&rcount, 1);
        if (c <= 20)
            fprintf(stderr, "[noabort] raise(SIGABRT) #%d intercepted — suspending thread\n", c);
        for (;;) pause();
    }
    // For other signals, call real raise
    if (!real_raise) {
        // Use syscall directly for non-SIGABRT
        return kill(getpid(), sig);
    }
    return real_raise(sig);
}

void __assert_fail(const char *e, const char *f, unsigned int l, const char *fn) {
    static int c = 0;
    int n = __sync_add_and_fetch(&c, 1);
    if (n <= 20)
        fprintf(stderr, "[noabort] assert(%s) at %s:%u ignored\n", e, f, l);
}
