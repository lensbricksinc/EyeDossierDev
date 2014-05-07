
#include "motionEstDS.h"
/*
% Computes motion vectors using Diamond Search method
%
% Based on the paper by Shan Zhu, and Kai-Kuang Ma
% IEEE Trans. on Image Processing
% Volume 9, Number 2, February 2000 :  Pages 287:290
%
% Input
%   imgP : The image for which we want to find motion vectors
%   imgI : The reference image
%   mbSize : Size of the macroblock
%   p : Search parameter  (read literature to find what this means)
%
% Ouput
%   motionVect : the motion vectors for each integral macroblock in imgP
%   DScomputations: The average number of points searched for a macroblock
%
% Written by Aroh Barjatya
*/


double costFuncMAD(cv::Mat a, cv::Mat b, int size)
{
    double val = 0;
#if 0
    // This is much worse performance-wise!!! Logically doesn't make  sense.
    cv::Mat c = cv::abs(a - b);
    val = cv::sum(c)[0,0];
#else
    for (int i=0; i<size; i++){
        for (int j=0; j<size; j++)
        {
            val += abs(a.at<float>(i,j) - b.at<float>(i,j));
        }
    
    }
#endif
    val = val/(double)(size*size);
    return val;
}

void motionEstDS(cv::Mat imgP, cv::Mat imgI, int mbSize, int p, double *motionVect[4], double &DScomputations){
    // imgP is of type CV_16S
    // imgI is of type CV_16S

    //cv::Size sizeImgI = imgI.size();
    //[row col] = size(imgI);
    int imgRows = imgI.rows;
    int imgCols = imgI.cols;

    int lenVectors = imgRows*imgCols/((mbSize*mbSize));
    int lenCosts = 9;

    for (int j=0; j<4; j++) {
        for (int i=0; i<lenVectors; i++)
        {
            motionVect[j][i] = 0.0;
        }
    }
   

    double *costs = new double[lenCosts];
    double *costs2 = new double[5];

    for (int i=0; i<lenCosts; i++)
        costs[i] = 65537.0;

    /*
    int L=0;
    float divisor=p+1;
    while (divisor>=2)
    {
        divisor= divisor/2;
        L=L+1;
    }
    */
    int L = (int)floor(log(p+1)/log(2));

    // The index points for Large Diamond search pattern
    int LDSP[9][2] = {{0, -2},
                      {-1, -1},
                      {1, -1},
                      {-2, 0},
                      {0, 0},
                      {2, 0},
                      {-1, 1},
                      {1, 1},
                      {0, 2}};

    // The index points for Small Diamond search pattern
    int SDSP[5][2] = {{0, -1},
                      {-1, 0},
                      {0, 0},
                      {1, 0},
                      {0, 1}};


    // we start off from the top left of the image
    // we will walk in steps of mbSize
    // for every marcoblock that we look at we will look for
    // a close match p pixels on the left, right, top and bottom of it

    int computations = 0;

    int mbCount = 0;
    int zeroThres = 0;
    double zeroCost  = 0.0;
    int point= 0;
    double cost=65536;

    for (int i=0; i<imgRows-mbSize+1; i+=mbSize)
    {
        for (int j=0; j<imgCols-mbSize+1; j+=mbSize)
        {
            // the Diamond search starts
            // we are scanning in raster order
            int x = j;      // 272
            int y = i;      // 0

            for (int k=0; k<lenCosts; k++)
                costs[k] = 65537.0;

            costs[4] = costFuncMAD(imgP(cv::Rect(j,i,mbSize,mbSize)), imgI(cv::Rect(j,i,mbSize,mbSize)), mbSize);
            //costs[4] = costFuncMAD(imgP(i:i+mbSize-1,j:j+mbSize-1), imgI(i:i+mbSize-1,j:j+mbSize-1),mbSize);
            zeroCost = costs[4];
            costs[4] = costs[4]-zeroThres;

            if (costs[4] < 0)
                costs[4] = 0;

            computations = computations + 1;

            // This is the first search so we evaluate all the 9 points in LDSP
            for (int k=0; k<9; k++)
            {
                int refBlkVer = y + LDSP[k][1];     // row/Vert co-ordinate for ref block   // -2   
                int refBlkHor = x + LDSP[k][0];     // col/Horizontal co-ordinate           // 272

                if ( refBlkVer < 0 || refBlkVer+mbSize > imgRows 
                    || refBlkHor < 0 || refBlkHor+mbSize > imgCols)
                {
                    continue;
                }

                if (k==4)
                    continue;

                costs[k] = costFuncMAD(imgP(cv::Rect(j,i,mbSize,mbSize)),
                                       imgI(cv::Rect(refBlkHor, refBlkVer,mbSize,mbSize)),
                                       mbSize);

                computations = computations + 1;
                //printf("Computation: FIRST LOOP\n");
            }

            // Find min cost and index
            cost= costs[0];
            point = 0;
            for (int k=1; k<lenCosts; k++)
            {
                if (costs[k]<cost)
                {
                    cost = costs[k];
                    point = k;
                }
            }


            // The SDSPFlag is set to 1 when the minimum
            // is at the center of the diamond     
            int SDSPFlag;
            int cornerFlag;
            int xLast, yLast;
            if (point == 4)
            {
                SDSPFlag = 1;
            }
            else
            {
                SDSPFlag = 0;

                if ( abs(LDSP[point][0]) == abs(LDSP[point][1]) )
                    cornerFlag = 0;
                else
                    cornerFlag = 1; // the x and y co-ordinates not equal on corners

                xLast = x;      // 272
                yLast = y;      // 0
                x = x + LDSP[point][0];
                y = y + LDSP[point][1];
                for (int k=0; k<lenCosts; k++)
                    costs[k] = 65537.0;

                costs[4] = cost;
            }

            while (SDSPFlag == 0)
            {
                if (cornerFlag == 1)
                {
                    for (int k=0; k<9; k++)
                    {
                        int refBlkVer = y + LDSP[k][1];   // row/Vert co-ordinate for ref block
                        int refBlkHor = x + LDSP[k][0];    // col/Horizontal co-ordinate
                        if ( refBlkVer < 0 || refBlkVer+mbSize > imgRows 
                           || refBlkHor < 0 || refBlkHor+mbSize > imgCols)
                        {
                            continue;
                        }

                        if (k == 4)
                            continue;

                        if ( refBlkHor >= xLast - 1  && refBlkHor <= xLast + 1 
                                && refBlkVer >= yLast - 1  && refBlkVer <= yLast + 1 )
                        {
                            continue;
                        }
                        else if (refBlkHor < j-p || refBlkHor > j+p || refBlkVer < i-p 
                                || refBlkVer > i+p)
                        {
                            continue;
                        }
                        else
                        {
                            costs[k] = costFuncMAD(imgP(cv::Rect(j,i,mbSize,mbSize)),
                                       imgI(cv::Rect(refBlkHor, refBlkVer,mbSize,mbSize)),
                                       mbSize);
                            computations = computations + 1;
                            //printf("Computation: SECOND LOOP\n");
                        }
                    

                    }
            
                }
                else
                {
                    switch(point)
                    {
                    case 1:
                    {
                        int refBlkVer = y + LDSP[0][1];    // row/Vert co-ordinate for ref block
                        int refBlkHor = x + LDSP[0][0];    // col/Horizontal co-ordinate
                        if ( refBlkVer < 0 || refBlkVer+mbSize > imgRows 
                            || refBlkHor < 0 || refBlkHor+mbSize > imgCols)
                        {
                            // do nothing, outside image boundary
                        }
                        else if (refBlkHor < j-p || refBlkHor > j+p || refBlkVer < i-p 
                                || refBlkVer > i+p)
                        {
                            // do nothing, outside search window
                        }
                        else
                        {
                            costs[0] = costFuncMAD(imgP(cv::Rect(j,i,mbSize,mbSize)),
                                       imgI(cv::Rect(refBlkHor,refBlkVer,mbSize,mbSize)),
                                       mbSize);
                            computations = computations + 1;
                            //printf("Computation: Case 1 (1)\n");
                        }
                                   
                        refBlkVer = y + LDSP[1][1];    // row/Vert co-ordinate for ref block
                        refBlkHor = x + LDSP[1][0];    // col/Horizontal co-ordinate
                        if ( refBlkVer < 0 || refBlkVer+mbSize > imgRows 
                            || refBlkHor < 0 || refBlkHor+mbSize > imgCols)
                        {
                            // do nothing, outside image boundary
                        }
                        else if (refBlkHor < j-p || refBlkHor > j+p || refBlkVer < i-p 
                                || refBlkVer > i+p)
                        {
                            // do nothing, outside search window
                        }
                        else
                        {
                            costs[1] = costFuncMAD(imgP(cv::Rect(j,i,mbSize,mbSize)),
                                       imgI(cv::Rect(refBlkHor,refBlkVer,mbSize,mbSize)),
                                       mbSize);
                            computations = computations + 1;
                            //printf("Computation: Case 1 (2)\n");
                        }
                        
                        refBlkVer = y + LDSP[3][1];    // row/Vert co-ordinate for ref block
                        refBlkHor = x + LDSP[3][0];    // col/Horizontal co-ordinate
                        if ( refBlkVer < 0 || refBlkVer+mbSize > imgRows
                            || refBlkHor < 0 || refBlkHor+mbSize > imgCols)
                        {
                            // do nothing, outside image boundary
                        }
                        else if (refBlkHor < j-p || refBlkHor > j+p || refBlkVer < i-p
                                || refBlkVer > i+p)
                        {
                            // do nothing, outside search window
                        }
                        else
                        {
                            costs[3] = costFuncMAD(imgP(cv::Rect(j,i,mbSize,mbSize)),
                                       imgI(cv::Rect(refBlkHor,refBlkVer,mbSize,mbSize)),
                                       mbSize);
                            computations = computations + 1;
                            //printf("Computation: Case 1 (3)\n");
                        }
                    }
                        break;
                
                    case 2:
                    {
                        int refBlkVer = y + LDSP[0][1];    // row/Vert co-ordinate for ref block
                        int refBlkHor = x + LDSP[0][0];    // col/Horizontal co-ordinate
                        if ( refBlkVer < 0 || refBlkVer+mbSize > imgRows 
                            || refBlkHor < 0 || refBlkHor+mbSize > imgCols)
                        {
                           // do nothing, outside image boundary
                        }
                        else if (refBlkHor < j-p || refBlkHor > j+p || refBlkVer < i-p 
                            || refBlkVer > i+p)
                        {
                            // do nothing, outside search window
                        }
                        else
                        {
                            costs[0] = costFuncMAD(imgP(cv::Rect(j,i,mbSize,mbSize)),
                                       imgI(cv::Rect(refBlkHor,refBlkVer,mbSize,mbSize)),
                                       mbSize);
                           computations = computations + 1;
                           //printf("Computation: Case 2 (1)\n");
                        }
                                   
                        refBlkVer = y + LDSP[2][1];   // row/Vert co-ordinate for ref block
                        refBlkHor = x + LDSP[2][0];   // col/Horizontal co-ordinate
                        if ( refBlkVer < 0 || refBlkVer+mbSize > imgRows
                            || refBlkHor < 0 || refBlkHor+mbSize > imgCols)
                        {
                           // do nothing, outside image boundary
                        }
                        else if (refBlkHor < j-p || refBlkHor > j+p || refBlkVer < i-p
                            || refBlkVer > i+p)
                        {
                            // do nothing, outside search window
                        }
                        else
                        {
                           costs[2] = costFuncMAD(imgP(cv::Rect(j,i,mbSize,mbSize)),
                                       imgI(cv::Rect(refBlkHor,refBlkVer,mbSize,mbSize)),
                                       mbSize);
                           computations = computations + 1;
                           //printf("Computation: Case 2 (2)\n");
                        }
                        
                        refBlkVer = y + LDSP[5][1];   // row/Vert co-ordinate for ref block
                        refBlkHor = x + LDSP[5][0];   // col/Horizontal co-ordinate
                        if ( refBlkVer < 0 || refBlkVer+mbSize > imgRows
                            || refBlkHor < 0 || refBlkHor+mbSize > imgCols)
                        {
                           // do nothing, outside image boundary
                        }
                        else if (refBlkHor < j-p || refBlkHor > j+p || refBlkVer < i-p 
                            || refBlkVer > i+p)
                        {
                            // do nothing, outside search window
                        }
                        else
                        {
                           costs[5] =  costFuncMAD(imgP(cv::Rect(j,i,mbSize,mbSize)),
                                       imgI(cv::Rect(refBlkHor,refBlkVer,mbSize,mbSize)),
                                       mbSize);
                           computations = computations + 1;
                           //printf("Computation: Case 2 (3)\n");
                        }
                    }
                        break;

                    case 6:
                    {
                        int refBlkVer = y + LDSP[3][1];   // row/Vert co-ordinate for ref block
                        int refBlkHor = x + LDSP[3][0];   // col/Horizontal co-ordinate
                        if ( refBlkVer < 0 || refBlkVer+mbSize > imgRows
                            || refBlkHor < 0 || refBlkHor+mbSize > imgCols)
                        {
                           // do nothing, outside image boundary
                        }
                        else if (refBlkHor < j-p || refBlkHor > j+p || refBlkVer < i-p
                            || refBlkVer > i+p)
                        {
                            // do nothing, outside search window
                        }
                        else
                        {
                           costs[3] = costFuncMAD(imgP(cv::Rect(j,i,mbSize,mbSize)),
                                       imgI(cv::Rect(refBlkHor,refBlkVer,mbSize,mbSize)),
                                       mbSize);
                           computations = computations + 1;
                           //printf("Computation: Case 6 (1)\n");
                        }
                                   
                        refBlkVer = y + LDSP[6][1];   // row/Vert co-ordinate for ref block
                        refBlkHor = x + LDSP[6][0];   // col/Horizontal co-ordinate
                        if ( refBlkVer < 0 || refBlkVer+mbSize > imgRows
                            || refBlkHor < 0 || refBlkHor+mbSize > imgCols)
                        {
                           // do nothing, outside image boundary
                        }
                        else if (refBlkHor < j-p || refBlkHor > j+p || refBlkVer < i-p 
                            || refBlkVer > i+p)
                        {
                            // do nothing, outside search window
                        }
                        else 
                        {
                           costs[6] = costFuncMAD(imgP(cv::Rect(j,i,mbSize,mbSize)),
                                       imgI(cv::Rect(refBlkHor,refBlkVer,mbSize,mbSize)),
                                       mbSize);
                           computations = computations + 1;
                           //printf("Computation: Case 6 (2)\n");
                        }
                        
                        refBlkVer = y + LDSP[8][1];   // row/Vert co-ordinate for ref block
                        refBlkHor = x + LDSP[8][0];   // col/Horizontal co-ordinate
                        if ( refBlkVer < 0 || refBlkVer+mbSize > imgRows 
                            || refBlkHor < 0 || refBlkHor+mbSize > imgCols)
                        {
                           // do nothing, outside image boundary
                        }
                        else if (refBlkHor < j-p || refBlkHor > j+p || refBlkVer < i-p 
                            || refBlkVer > i+p)
                        {
                            // do nothing, outside search window
                        }
                        else
                        {
                           costs[8] = costFuncMAD(imgP(cv::Rect(j,i,mbSize,mbSize)),
                                       imgI(cv::Rect(refBlkHor,refBlkVer,mbSize,mbSize)),
                                       mbSize);
                           computations = computations + 1;
                           //printf("Computation: Case 6 (3)\n");
                        }
                    }
                        break;

                    case 7:
                    {
                        int refBlkVer = y + LDSP[5][1];   // row/Vert co-ordinate for ref block
                        int refBlkHor = x + LDSP[5][0];   // col/Horizontal co-ordinate
                        if ( refBlkVer < 0 || refBlkVer+mbSize > imgRows
                            || refBlkHor < 0 || refBlkHor+mbSize > imgCols)
                        {
                           // do nothing, outside image boundary
                        }
                        else if (refBlkHor < j-p || refBlkHor > j+p || refBlkVer < i-p
                            || refBlkVer > i+p)
                        {
                            // do nothing, outside search window
                        }
                        else 
                        {
                           costs[5] = costFuncMAD(imgP(cv::Rect(j,i,mbSize,mbSize)),
                                       imgI(cv::Rect(refBlkHor,refBlkVer,mbSize,mbSize)),
                                       mbSize);
                           computations = computations + 1;
                           //printf("Computation: Case 7 (1)\n");
                        }
                                   
                        refBlkVer = y + LDSP[7][1];   // row/Vert co-ordinate for ref block
                        refBlkHor = x + LDSP[7][0];   // col/Horizontal co-ordinate
                        if ( refBlkVer < 0 || refBlkVer+mbSize > imgRows
                            || refBlkHor < 0 || refBlkHor+mbSize > imgCols)
                        {
                           // do nothing, outside image boundary
                        }
                        else if (refBlkHor < j-p || refBlkHor > j+p || refBlkVer < i-p
                            || refBlkVer > i+p)
                        {
                            // do nothing, outside search window
                        }
                        else
                        {
                           costs[7] = costFuncMAD(imgP(cv::Rect(j,i,mbSize,mbSize)),
                                       imgI(cv::Rect(refBlkHor,refBlkVer,mbSize,mbSize)),
                                       mbSize);
                           computations = computations + 1;
                           //printf("Computation: Case 7 (2)\n");
                        }
                        
                        refBlkVer = y + LDSP[8][1];   // row/Vert co-ordinate for ref block
                        refBlkHor = x + LDSP[8][0];   // col/Horizontal co-ordinate
                        if ( refBlkVer < 0 || refBlkVer+mbSize > imgRows
                            || refBlkHor < 0 || refBlkHor+mbSize > imgCols)
                        {
                           // do nothing, outside image boundary
                        }
                        else if (refBlkHor < j-p || refBlkHor > j+p || refBlkVer < i-p 
                            || refBlkVer > i+p)
                        {
                            // do nothing, outside search window
                        }
                        else
                        {
                           costs[8] = costFuncMAD(imgP(cv::Rect(j,i,mbSize,mbSize)),
                                       imgI(cv::Rect(refBlkHor,refBlkVer,mbSize,mbSize)),
                                       mbSize);
                           computations = computations + 1;
                           //printf("Computation: Case 7 (3)\n");
                        }
                    }
                        break;


                    default:
                        {
                        break;
                        }
                    }
                }
            
                // Find min cost and index
                cost= costs[0];
                point = 0;
                for (int k=1; k<lenCosts; k++)
                {
                    if (costs[k]<cost)
                    {
                        cost = costs[k];
                        point = k;
                    }
                }

                if (point == 4)
                {
                    SDSPFlag = 1;
                }
                else
                {
                    SDSPFlag = 0;
                    if ( abs(LDSP[point][0]) == abs(LDSP[point][1]) )
                    {
                        cornerFlag = 0;
                    }
                    else
                    {
                        cornerFlag = 1;
                    }

                    xLast = x;  // 274
                    yLast = y;  // 0
                    x = x + LDSP[point][0];
                    y = y + LDSP[point][1];

                    for (int i=0; i<lenCosts; i++)
                        costs[i] = 65537;

                    costs[4] = cost;
                }

            }   // while (SDSPFlag == 0)
        


            for (int t=0; t<5; t++)
                costs2[t]=65537.0;

            costs2[2] = cost;

            for (int k=0; k<5; k++)
            {
                int refBlkVer = y + SDSP[k][1];   // row/Vert co-ordinate for ref block
                int refBlkHor = x + SDSP[k][0];   // col/Horizontal co-ordinate
                if ( refBlkVer < 0 || refBlkVer+mbSize > imgRows
                  || refBlkHor < 0 || refBlkHor+mbSize > imgCols)
                {
                    continue; // do nothing, outside image boundary
                }
                else if (refBlkHor < j-p || refBlkHor > j+p || refBlkVer < i-p
                            || refBlkVer > i+p)
                {
                    continue;   // do nothing, outside search window
                }
            
                if (k == 2)
                    continue;

                costs2[k] = costFuncMAD(imgP(cv::Rect(j,i,mbSize,mbSize)),
                                       imgI(cv::Rect(refBlkHor,refBlkVer,mbSize,mbSize)),
                                       mbSize);
                computations = computations + 1;
                //printf("Computation: LAST\n");

            }
        
            cost=costs2[0];
            point= 0;
            for (int k=1; k<5; k++)
            {
                if (costs2[k]<cost)
                {
                    cost = costs2[k];
                    point = k;
                }
            }
        
            x = x + SDSP[point][0];
            y = y + SDSP[point][1];

            if (lenVectors<=mbCount)
                printf("Size of lenVectors is greater than mbCount.\n");
        
            motionVect[0][mbCount] = y - i;    // row co-ordinate for the vector
            motionVect[1][mbCount] = x - j;    // col co-ordinate for the vector            
            motionVect[2][mbCount] = cost;
            motionVect[3][mbCount] = zeroCost; // cost at zero MV
            mbCount = mbCount + 1;
        }

    }

    delete[] costs2;
    costs2 = nullptr;
    delete[] costs;
    costs = nullptr;

    DScomputations = computations/(mbCount - 1);
}

    


   
