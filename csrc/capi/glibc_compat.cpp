// glibc 2.38+ headers redirect strtol/strtoul to __isoc23_* (C23 binary-literal
// parsing). Those symbols carry a GLIBC_2.38 version requirement, which makes the
// resulting .so unloadable on older runtimes (Debian 12 / glibc 2.36, e.g. the
// Modal T4 images). Nothing in this library depends on C23 base-prefix handling,
// so we define the __isoc23_* entry points locally and forward them to the
// classic, always-present strtol/strtoul.
//
// The __asm__ labels bind the forwarding calls to the unversioned-default
// `strtol`/`strtoul` symbols (GLIBC_2.2.5) rather than re-entering the header
// redirect, which would recurse infinitely.

extern "C" {

extern long __compat_strtol(const char *, char **, int) __asm__("strtol");
extern unsigned long __compat_strtoul(const char *, char **, int) __asm__("strtoul");
extern long long __compat_strtoll(const char *, char **, int) __asm__("strtoll");
extern unsigned long long __compat_strtoull(const char *, char **, int) __asm__("strtoull");

long __isoc23_strtol(const char *nptr, char **endptr, int base) {
  return __compat_strtol(nptr, endptr, base);
}

unsigned long __isoc23_strtoul(const char *nptr, char **endptr, int base) {
  return __compat_strtoul(nptr, endptr, base);
}

long long __isoc23_strtoll(const char *nptr, char **endptr, int base) {
  return __compat_strtoll(nptr, endptr, base);
}

unsigned long long __isoc23_strtoull(const char *nptr, char **endptr, int base) {
  return __compat_strtoull(nptr, endptr, base);
}

} // extern "C"
