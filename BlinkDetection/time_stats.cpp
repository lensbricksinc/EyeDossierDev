#include "time_stats.h"
#include <Windows.h>

TIMETRACKER::TIMETRACKER(int length)
{
    if (length > 0)
        mLength = length;
    else
        mLength = 1;

    mDurations = new unsigned long[mLength];
    resetStats();
}


void TIMETRACKER::addNewEntry()
{
    unsigned long currentTime = GetTickCount();
    if (lastTimeEntry == 0)
    {
        lastTimeEntry = currentTime;
        return;
    }

    mDurations[mCurrPosition] = currentTime - lastTimeEntry;
    
    if (mNumEntries < mLength)
        mNumEntries++;

    mCurrPosition++;
    if (mCurrPosition == mLength)
        mCurrPosition = 0;

    lastTimeEntry = currentTime;
}


double TIMETRACKER::getFPS()
{
    unsigned long average= 0;
    for (int i=0; i<mNumEntries; i++)
    {
        average += mDurations[i];
    }

    double dur = (double)average;
    dur = dur/(double)mNumEntries;
    if (dur >0)
        return 1000/dur;
    else
        return 0.0;
}


void TIMETRACKER::resetStats()
{
    mNumEntries= 0;
    lastTimeEntry = 0;
    mCurrPosition = 0;
}

