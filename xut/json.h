#pragma once

#ifdef string_t
#undef string_t
#endif
#include"nlohmann/json.hpp"
#undef string_t
#include<fstream>
#include<iomanip>
#include"def.h"

_XUT_BEG

template<typename _ValT, typename _JElemT>
inline void jsonGetArray(_JElemT& jx, _ValT* val, int dsize = 0)
{
	auto v = jx.get<std::vector<_ValT>>();
	CV_Assert(dsize == 0 || v.size() == dsize);
	if (!v.empty())
		memcpy(val, &v[0], sizeof(_ValT) * v.size());
}

inline nlohmann::json jsonLoad(const std::string& file)
{
	std::ifstream istream(file);
	if (!istream)
		throw "file open failed";

	nlohmann::json jf;
	istream >> jf;
	return jf;
}

template<typename _JsonT>
inline void jsonSave(_JsonT& jf, const std::string& file, int w = 0)
{
	std::ofstream  os(file);
	if (!os)
		throw "file open failed";
	os << std::setw(w) << jf;
}


_XUT_END

