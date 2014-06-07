#ifndef __GETFILENAMES_H__
#define __GETFILENAMES_H__

#include <iostream>
#include <vector>
#include <Windows.h>
//using namespace std;

std::string wchar_t2string(const wchar_t *wchar);

wchar_t *string2wchar_t(const std::string &str);

std::vector<std::string> listFilesInDirectory(std::string directoryName);

#endif