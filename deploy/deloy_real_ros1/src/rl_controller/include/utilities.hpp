#pragma once

template <typename T>
T crcCore(T* ptr, T len) 
{
	T xbit = 0;
	T data = 0;
	T CRC32 = 0xFFFFFFFF;
	const T dwPolynomial = 0x04c11db7;
	for (T i = 0; i < len; i++) 
	{
		xbit = 1 << 31;
		data = ptr[i];
		for (T bits = 0; bits < 32; bits++) 
		{
			if (CRC32 & 0x80000000) 
			{
				CRC32 <<= 1;
				CRC32 ^= dwPolynomial;
			} 
			else 
			{
				CRC32 <<= 1;
			}
			if (data & xbit)
			{
				CRC32 ^= dwPolynomial;
			}
			xbit >>= 1;
		}
	}
	return CRC32;
}
