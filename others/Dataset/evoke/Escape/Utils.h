#ifndef ESCAPE_UTILS_H_
#define ESCAPE_UTILS_H_

#include <algorithm>

namespace Escape
{

//x choose n.
template <int n, typename T>
T choose(T x)
{
  T num = 1;
  T den = x;
  for (int i = 1; i < n; ++i)
  {
    num *= (i + 1); //this can easily overflow!!!
    den *= (x - i);
  }
  
  return den / num;
}

template <typename T>
T choosec(T n, int k)
{
  if (n < 0)
  	return 0;
  if (k > n)
  	return 0;
  if (k == 0)
  	return 1;
  T nom = n;
  T denom = 1;
  for (int i = 1; i < k; i++)
  {
    nom *= (n-i); //this can easily overflow!!!
    denom *= (i+1);
  }
  return nom / denom;
}

//max(0, x - y)
//Safe to use with unsigned types
template <typename T>
T czsub(T x, T y)
{
  return x > y ? x - y : 0;
}



}
#endif
