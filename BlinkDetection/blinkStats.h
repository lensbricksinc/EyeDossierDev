#ifndef __BLINK_STATS__
#define __BLINK_STATS__

#include <fstream>

class BLINK_STATS{

private:
    int abc;
    void getCurrentTime();
    int startDate;
    int startMonth;
    int startHour;
    int startMinutes;
    int startSeconds;
    std::ofstream outFile;
    time_t startTime;

public:
    BLINK_STATS();
    ~BLINK_STATS();
    enum BLINK_STATS_EVENT
    {
        EVENT_BLINK_NODATA = 0,
        EVENT_BLINK_PAUSED = 1,
        EVENT_BLINK_ANALYSING = 2,
        EVENT_BLINK_EYEBLINK = 3
    };



    void addEvent(BLINK_STATS::BLINK_STATS_EVENT blinkEvent);

};

#endif
