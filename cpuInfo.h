WeChat: cstutorcs
QQ: 749389476
Email: tutorcs@163.com
#ifdef _WIN32
#include <intrin.h>
#else
#include <cpuid.h>
#endif

//Struct for reading info about what features the CPU supports
struct cpuINFO
{
	char brand[0x40];
	char brandHex[0x81];
	int avx = 0;
	int avx2 = 0;
	int fma3 = 0;
	int avx512f = 0;
};

//Get info about the CPU such as name, brand and instruction sets supported (e.g. avx, avx2, avx512)
int cpuInfoGet(cpuINFO* c)
{

	for (int i = 0; i < 0x40; i++)
	{
		c[0].brand[i] = 0;
	}
	for (int i = 0; i < 0x81; i++)
	{
		c[0].brandHex[i] = 0;
	}

	int a = 0x80000000;
	int fma3 = 0;
	int avx = 0;
	int avx2 = 0;
	int avx512f = 0;

	//Unforunately this part is platform dependent. Calls the cpuID instruction to find out what is supported.
	//https://en.wikipedia.org/wiki/CPUID
#ifdef _WIN32
	int cpuInfo[4];

	__cpuid((int*)&cpuInfo[0], a);

	if (cpuInfo[0] >= 0x80000004)
	{
		__cpuid((int*)&c[0].brand[0], 0x80000002);
		__cpuid((int*)&c[0].brand[16], 0x80000003);
		__cpuid((int*)&c[0].brand[32], 0x80000004);
	}
	a = 1;
	__cpuidex((int*)&cpuInfo[0], a, 0);
	fma3 = cpuInfo[2] >> 12 & 1;
	avx = cpuInfo[2] >> 28 & 1;
	a = 7;
	__cpuidex((int*)&cpuInfo[0], a, 0);
	avx2 = cpuInfo[1] >> 5 & 1;
	avx512f = cpuInfo[1] >> 16 & 1;
#else
	unsigned int cpuInfo[4];

	__get_cpuid(a, &cpuInfo[0], &cpuInfo[1], &cpuInfo[2], &cpuInfo[3]);

	if (cpuInfo[0] >= 0x80000004)
	{
		//cpuidGet(0x80000002,((cpuidType)&brand[0]));
		__get_cpuid(0x80000002, (unsigned int*)&c[0].brand[0], (unsigned int*)&c[0].brand[4], (unsigned int*)&c[0].brand[8], (unsigned int*)&c[0].brand[12]);
		__get_cpuid(0x80000003, (unsigned int*)&c[0].brand[16], (unsigned int*)&c[0].brand[20], (unsigned int*)&c[0].brand[24], (unsigned int*)&c[0].brand[28]);
		__get_cpuid(0x80000004, (unsigned int*)&c[0].brand[32], (unsigned int*)&c[0].brand[36], (unsigned int*)&c[0].brand[40], (unsigned int*)&c[0].brand[44]);
	}
	a = 1;
	//__get_cpuid(a, &cpuInfo[0], &cpuInfo[1], &cpuInfo[2], &cpuInfo[3]);
	__cpuid_count(a, 0, cpuInfo[0], cpuInfo[1], cpuInfo[2], cpuInfo[3]);
	fma3 = cpuInfo[2] >> 12 & 1;
	avx = cpuInfo[2] >> 28 & 1;
	a = 7;
	//__get_cpuid(a, &cpuInfo[0], &cpuInfo[1], &cpuInfo[2], &cpuInfo[3]);
	__cpuid_count(a, 0, cpuInfo[0], cpuInfo[1], cpuInfo[2], cpuInfo[3]);
	avx2 = cpuInfo[1] >> 5 & 1;
	avx512f = cpuInfo[1] >> 16 & 1;
#endif

	//The brand of the CPU as hexadecimal (rather than ascii). This hex string version will be used as a 'unique' identifier for things like writing the FFTW wisdom to file.
	for (int i = 0; i < 0x40; i++)
	{
		sprintf(&c[0].brandHex[2 * i], "%02x", c[0].brand[i]);
	}

	c[0].avx = avx;
	c[0].fma3 = fma3;
	c[0].avx2 = avx2;
	c[0].avx512f = avx512f;

	return 1;
}
