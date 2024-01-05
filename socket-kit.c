#include <errno.h>
#include <sys/socket.h>

int connect(int fd, const struct sockaddr *addr, socklen_t len)
{
    errno = ENETDOWN;
    return -1;
}