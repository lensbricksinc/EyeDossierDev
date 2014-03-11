
#include "getFileNames.h"


std::string wchar_t2string(const wchar_t *wchar)
{
    std::string str = "";
    int index = 0;
    while(wchar[index] != 0)
    {
        str += (char)wchar[index];
        ++index;
    }
    return str;
}

wchar_t *string2wchar_t(const std::string &str)
{
    wchar_t wchar[260];
    int index = 0;
    while(index < str.size())
    {
        wchar[index] = (wchar_t)str[index];
        ++index;
    }
    wchar[index] = 0;
    return wchar;
}

std::vector<std::string> listFilesInDirectory(std::string directoryName)
{
    WIN32_FIND_DATA FindFileData;
    wchar_t * FileName = string2wchar_t(directoryName);
    HANDLE hFind = FindFirstFile(FileName, &FindFileData);

    std::vector<std::string> listFileNames;
    listFileNames.push_back(wchar_t2string(FindFileData.cFileName));

    while (FindNextFile(hFind, &FindFileData))
        listFileNames.push_back(wchar_t2string(FindFileData.cFileName));

    return listFileNames;
}


