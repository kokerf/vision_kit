#ifndef _MATH_HPP_
#define _MATH_HPP_

#include <cmath>
#include <cstdlib>
#include <stdint.h>

#define VK_EPS 1e-20

#define VK_DESCALE(x,n) (((x) + (1 << ((n)-1))) >> (n))

#define VK_MIN(x,y) ((x>y)?y:x)
#define VK_MAX(x,y) ((x>y)?x:y)

namespace vk{

inline double Rand(double min, double max)
{ return (((double)rand()/((double)RAND_MAX + 1.0)) * (max - min + 1)) + min;}

inline int Rand(int min, int max)
{ return (((double)rand()/((double)RAND_MAX + 1.0)) * (max - min + 1)) + min;}

inline void sampleNpoints(const int& min, const int& max, const int& num, int* out)
{
    int* store = new int[num];
    for(int i = 0; i < num; ++i)
    {
        const int maxi = max - i;
        out[i] = vk::Rand(min, maxi);
        store[i] = maxi;

        for(int j = 0; j < i; j++)
        {
            if(out[i] == out[j])
            {
                out[i] = store[j];
                store[j] = maxi;
            }
        }
    }
    delete[] store;
}

}//! vk

#endif