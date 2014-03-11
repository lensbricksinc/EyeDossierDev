#include "blinkStats.h";
#include <time.h>
#include "Windows.h"

BLINK_STATS::BLINK_STATS()
{
    getCurrentTime();
    outFile.open("stats.txt");
    outFile << "Current Time: "<<std::endl<<std::endl;
    outFile.flush();
}

BLINK_STATS::~BLINK_STATS()
{
    outFile.flush();
    outFile.close();
}


void BLINK_STATS::getCurrentTime()
{

    time_t clock;
    struct tm y2k;

    double seconds;

    y2k.tm_hour = 0;   y2k.tm_min = 0; y2k.tm_sec = 0;
    y2k.tm_year = 114; y2k.tm_mon = 1; y2k.tm_mday = 22;

    time(&clock);  /* get current time; same as: timer = time(NULL)  */

    startTime = clock;
    seconds = difftime(clock, mktime(&y2k));

    int additionalHours = (int)(seconds / (60 * 60));
    int additionalDay = additionalHours / 24;

    startDate = y2k.tm_mday + additionalDay;
    startMonth = y2k.tm_mon;
    startHour = additionalHours % 24;
    int temp = (int)(seconds / 60);
    startMinutes = temp % 60;
    temp = (int)(seconds / 60);
    startSeconds = (int)(seconds - temp*60);

    //printf("%.f seconds since %d/ %d/ %d in the current timezone", seconds, y2k.tm_mday, y2k.tm_mon = 1, 1900 + 114);


};



enum BLINK_STATS_EVENT
{
    EVENT_BLINK_NODATA = 0,
    EVENT_BLINK_PAUSED = 1,
    EVENT_BLINK_ANALYSING = 2,
    EVENT_BLINK_EYEBLIN = 3
};



void BLINK_STATS::addEvent(BLINK_STATS::BLINK_STATS_EVENT blinkEvent)
{
    time_t clock;

    double seconds;

    time(&clock); 
    seconds = difftime(clock, startTime);

    //unsigned long currentTime = GetTickCount();

    outFile << "Event: " << blinkEvent << ". Time: " << seconds << std::endl;
    outFile.flush();
};

