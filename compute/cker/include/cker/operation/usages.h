#include <cstdio>
#include <cstring>
#include <thread>
#include <unistd.h>
#include <ctime>

#include <sys/time.h>
#include <sys/resource.h>


inline int getRSS()
{
  rusage used;
  if (0 == getrusage(RUSAGE_SELF, &used))
  {
      return used.ru_isrss;
  }
  else 
    return 0;
}


inline int getMemoryUsage()
{
  int pid = getpid();
  char target[30], buf[4096];

  // printf("\n%d\n", pid);

  sprintf(target, "/proc/%d/status", pid);
  FILE *f = fopen(target, "r");
  fread(buf, 1, 4095, f);
  buf[4095] = '\0';
  fclose(f);

  int mem;
  char *ptr = strstr(buf, "VmSize:");
  sscanf(ptr, "%*s %d", &mem);
  
  return mem;
}