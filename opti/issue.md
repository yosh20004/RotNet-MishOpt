## A little qustion about code in avx_mathfun.h ##

Hi, i have something confused about the code in avx_mathfun.h
In the function **v8sf exp256_ps(v8sf x)**, lines 266-273, your code is
```cpp
  tmp = _mm256_floor_ps(fx);

  /* if greater, substract 1 */
  //v8sf mask = _mm256_cmpgt_ps(tmp, fx);    
  v8sf mask = _mm256_cmp_ps(tmp, fx, _CMP_GT_OS);    
  mask = _mm256_and_ps(mask, one);
  fx = _mm256_sub_ps(tmp, mask);
```

but its seems that tmp will not greater than fx, so i think element in mask wont be 1, so is it right to change code to
```cpp
  tmp = _mm256_floor_ps(fx);                                
  /* if greater, substract 1 */
  //v8sf mask = _mm256_cmpgt_ps(tmp, fx);    
  //   v8sf mask = _mm256_cmp_ps(tmp, fx, _CMP_GT_OS);
  //   mask = _mm256_and_ps(mask, one);
  //   fx = _mm256_sub_ps(tmp, mask);
  fx = tmp;
``` 

sorry for bother you, thanks for your work it help me a lot