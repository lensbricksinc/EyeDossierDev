
#ifndef __TIME_STATS_H__
#define __TIME_STATS_H__

class TIMETRACKER {
private:
    int mLength;
    // start and end times
    unsigned long* mDurations;
    unsigned long lastTimeEntry;
    int mNumEntries;
    int mCurrPosition;

public:
    TIMETRACKER(int length);

    void addNewEntry();

    double getFPS();

    void resetStats();
};

#endif
