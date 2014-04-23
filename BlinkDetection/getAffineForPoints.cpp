
#include "getAffineForPoints.h"



void getAffineForPoints(std::vector<cv::Point2f> &points1,
                            std::vector<cv::Point2f> &points2,
                            std::vector<unsigned int> &inliers,
                            cv::Mat &outAffMatx)
{
  int sampleSize = 2;
  assert(points1.size() == points2.size());
  int totalPoints = points1.size();
  int maxUniqueVals;
  if (totalPoints > 2)
    maxUniqueVals = (totalPoints * (totalPoints-1) * (totalPoints-2))/6;    // nC3
  else
    maxUniqueVals = 1;
  int MAX_ITER = MIN(100, maxUniqueVals);
  int currIter = 0;
  cv::Mat bestAffineMat;
  int numTrials = MAX_ITER;
  int threshDist = 4;
  double bestDis = threshDist*totalPoints + 1;    // +4 to tackle the case when all points are outliers

  /*
  if (frameCount == 35)
  {
    frameCount = 35;
  }
  */
  //printf("\n\n New Trial= %d, ", numTrials);
  while (currIter < numTrials)
  {
    // Take 2 to 3 points at random, and calculate affine parameters
    // Ensure that the points are unique
    int currSetOfPoints[3] = {rand()%totalPoints, rand()%totalPoints, rand()%totalPoints};

    while (currSetOfPoints[0] == currSetOfPoints[1])
    {
      currSetOfPoints[1] = rand()%totalPoints;
    }

    while (currSetOfPoints[0] == currSetOfPoints[2] || currSetOfPoints[1] == currSetOfPoints[2])
    {
      currSetOfPoints[2] = rand()%totalPoints;
    }

    //currSetOfPoints[0] = 2;
    //currSetOfPoints[1] = 0;

    std::vector<cv::Point2f> points1Sub = std::vector<cv::Point2f>();
    std::vector<cv::Point2f> points2Sub = std::vector<cv::Point2f>();
    std::vector<cv::Point2f> points2Derived = std::vector<cv::Point2f>();
    points1Sub.push_back(points1[currSetOfPoints[0]]);
    points1Sub.push_back(points1[currSetOfPoints[1]]);
    //points1Sub.push_back(points1[currSetOfPoints[2]]);

    points2Sub.push_back(points2[currSetOfPoints[0]]);
    points2Sub.push_back(points2[currSetOfPoints[1]]);
    //points2Sub.push_back(points2[currSetOfPoints[2]]);

    // Get affine transformation parameters
    //cv::Mat affineMat = getAffineTransform(points1Sub, points2Sub);   // affineMat is of type CV_64F (== 6)

    std::vector<int> indices;
    indices.push_back(-1);
    cv::Mat affineMat =  computeTForm(2, 
                                      points1Sub,
                                      points2Sub,
                                      indices);

    transformPointsForward(points1, points2Derived, affineMat);
    //cv::warpAffine(points1Sub, points2SubDerived, affineMat, cv::Size(0,0));    // works only for images
    double dist= 0.0;
    int index =0;
    
    while (dist < bestDis && index < totalPoints)
    {
      dist += MIN(cv::norm(points2Derived[index]-points2[index]), threshDist);
      index++;
    }

    if (dist < bestDis)
    {
      bestDis = dist;
      bestAffineMat = affineMat;
      int confidence = 99;
      int inlierNum = totalPoints - (int)(bestDis/threshDist);
      int num = computeLoopNumberSVD(sampleSize, confidence, totalPoints, inlierNum);
      //numTrials = MIN(numTrials, num);
      //printf("%d, ", numTrials);
    }
    currIter++;
  }
  //printf("\n", numTrials);
  // We have the best AffineTransform Matrix now. Calculate outliers now
  inliers = std::vector<unsigned int>();
  {
    std::vector<cv::Point2f> points2Derived = std::vector<cv::Point2f>();
    transformPointsForward(points1, points2Derived, bestAffineMat);

    for (int i=0; i<totalPoints; i++)
    {
      int d = cv::norm(points2Derived[i]-points2[i]);
      if (d < threshDist)
        inliers.push_back(1);
      else
        inliers.push_back(0);
    }
  }
  
  // Recalculate affine matrix using only inliers
  std::vector<int> indices = std::vector<int>();
  for (int i=0; i<totalPoints; i++)
  {
    if (inliers[i] == 1)
      indices.push_back(i);
  }


  cv::Mat affineMatFinal =  computeTForm(2, 
                                      points1,
                                      points2,
                                      indices);
  //std::cout << affineMatFinal << std::endl;

  outAffMatx = affineMatFinal;
  return;
}


void drawFaceBox(cv::Mat frame, std::vector<cv::Point2f> faceBox)
{
  std::vector<cv::Point2i> faceBoxInt = std::vector<cv::Point2i>();

  for (int i=0; i< faceBox.size(); i++)
    faceBoxInt.push_back(cv::Point2i(faceBox[i]));    // Verified that rounding of values is being done here.

  cv::polylines(frame, faceBoxInt, true, cv::Scalar(255,0,255),2);
}

cv::Rect findFaceBox(cv::Mat frame, cv::Rect origSizeFaceBox, bool &outBoolForceReinit)
{
  cv::Rect retValue = cv::Rect(-1,-1,-1,-1);
  int xMin=frame.cols;
  int yMin=frame.rows;
  int xMax= 0;
  int yMax= 0;
  outBoolForceReinit = false;

  for (int i=0; i<frame.rows; i++)
  {
    for (int j=0; j<frame.cols; j++) {
       if (frame.at<cv::Vec3b>(i,j)[0] == 255
         && frame.at<cv::Vec3b>(i,j)[1] == 0
         && frame.at<cv::Vec3b>(i,j)[2] == 255)
       {
          if (xMin > j)
            xMin = j;
          if (yMin > i)
            yMin = i;
          if (xMax< j)
            xMax = j;
          if (yMax< i)
            yMax = i;
       }
    }
  }
  retValue = cv::Rect(xMin, yMin, xMax-xMin+1, yMax-yMin+1);    // This will always be in range of the frame

  if (retValue.width != origSizeFaceBox.width)
  {
    int diff = retValue.width - origSizeFaceBox.width;
    int diff_2 = diff/2;
    retValue.x = retValue.x + diff_2;
    retValue.width = origSizeFaceBox.width;
    if (retValue.x < 0 || (retValue.x + retValue.width >= frame.cols))
        outBoolForceReinit = true;
  }

  if (retValue.height != origSizeFaceBox.height)
  {
    int diff = retValue.height - origSizeFaceBox.height;
    int diff_2 = diff/2;
    retValue.y = retValue.y + diff_2;
    retValue.height = origSizeFaceBox.height;
    if (retValue.y < 0 || (retValue.y + retValue.height >= frame.rows))
        outBoolForceReinit = true;
  }

  return retValue;
};

bool isFeatureResetRequired(cv::Mat fullFrame, cv::vector<cv::Point2f> faceBox, std::vector<cv::Point2f> featurePoints)
{
  bool retValue = false;
  std::vector<float> dist[4];
  cv::Point2f point0, point1, point2, point3;
  float areaFaceBox, areaCoveredByPoints;

  // Find distance of points from the four corner points. Use the closest points, and calculate the area covered by it.
  assert (faceBox.size() == 4);

  if (featurePoints.size() <4)
  {
    retValue = true;
    goto EXIT;
  }

  point0 = faceBox[0];
  point1 = faceBox[1];
  point2 = faceBox[2];
  point3 = faceBox[3];

  for (int i=0; i<featurePoints.size(); i++)
  {
    cv::Point2f currPoint = featurePoints[i];
    dist[0].push_back(cv::norm(currPoint - point0));
    dist[1].push_back(cv::norm(currPoint - point1));
    dist[2].push_back(cv::norm(currPoint - point2));
    dist[3].push_back(cv::norm(currPoint - point3));
  }

  {
    float minDist[4]= {dist[0][0], dist[1][0], dist[2][0], dist[3][0]};
    int minDistIndex[4] = {0, 0, 0, 0};
    for (int i=1; i<featurePoints.size(); i++)
    {
      if (dist[0][i] < minDist[0]) {minDist[0] = dist[0][i]; minDistIndex[0]= i;}
      if (dist[1][i] < minDist[1]) {minDist[1] = dist[1][i]; minDistIndex[1]= i;}
      if (dist[2][i] < minDist[2]) {minDist[2] = dist[2][i]; minDistIndex[2]= i;}
      if (dist[3][i] < minDist[3]) {minDist[3] = dist[3][i]; minDistIndex[3]= i;}
    }

    // Given that co-ordinates of a polygon are arranged in counter-clockwise/clockwise fashion, it's area is given by
    // (x1*y2-y1*x2)+ (x2*y3-y2*x3) ... + (xn*y1-yn*x1)/2

    areaFaceBox = ((point0.x*point1.y - point0.y*point1.x) + (point1.x*point2.y - point1.y*point2.x) 
                            + (point2.x*point3.y - point2.y*point3.x) + (point3.x*point0.y - point3.y*point0.x))/2;

    point0 = featurePoints[minDistIndex[0]];
    point1 = featurePoints[minDistIndex[1]];
    point2 = featurePoints[minDistIndex[2]];
    point3 = featurePoints[minDistIndex[3]];

    areaCoveredByPoints = ((point0.x*point1.y - point0.y*point1.x) + (point1.x*point2.y - point1.y*point2.x) 
                            + (point2.x*point3.y - point2.y*point3.x) + (point3.x*point0.y - point3.y*point0.x))/2;
  
    if ((areaCoveredByPoints/areaFaceBox) < 0.15)
      retValue = true;

  }

EXIT:
  return retValue;
}



void computerSimilaritySVD(std::vector<cv::Point2f> &points1,
                            std::vector<cv::Point2f> &points2,
                            cv::Mat &outAffMatx)
{
  assert(points1.size() == points2.size());
  int numPoints = points1.size();
  int index=0;
  
  cv::Mat matConstraints (2*numPoints, 5, CV_32F);
  index=0;
  for (int i=0; i<2*numPoints; i+=2)
  {
    matConstraints.at<float>(i,0) = -points1[index].y;
    matConstraints.at<float>(i,1) = points1[index].x;
    matConstraints.at<float>(i,2) = 0;
    matConstraints.at<float>(i,3) = -1;
    matConstraints.at<float>(i,4) = points2[index].y;
    index++;
  }

  index=0;
  for (int i=1; i<2*numPoints; i+=2)
  {
    matConstraints.at<float>(i,0) = points1[index].x;
    matConstraints.at<float>(i,1) = points1[index].y;
    matConstraints.at<float>(i,2) = 1;
    matConstraints.at<float>(i,3) = 0;
    matConstraints.at<float>(i,4) = -points2[index].x;
    index++;
  }

  //std::cout << matConstraints << std::endl;

  cv::SVD svdA(matConstraints, cv::SVD::FULL_UV);
  //std::cout << svdA.u.t() * cv::Mat(matConstraints) << std::endl;
  //std::cout << svdA.u << std::endl;
  //std::cout << std::endl << std::endl << svdA.w << std::endl << std::endl;
  

  cv::Mat v = svdA.vt.t();          // This is required because cv::SVD returnds v-transpose by default.
  //std::cout << v << std::endl;
  v = v/v.at<float>(4,4);

  //std::cout << matConstraints * svdA.vt.t() << std::endl << std::endl;

  /*
  printf("Type of V= %d\n", v.type());
  */

  outAffMatx = cv::Mat(3,3, CV_32F);
  outAffMatx.at<float>(0,0) = v.at<float>(0,4);
  outAffMatx.at<float>(1,0) = v.at<float>(1,4);
  outAffMatx.at<float>(2,0) = v.at<float>(2,4);
  outAffMatx.at<float>(0,1) = - v.at<float>(1,4);
  outAffMatx.at<float>(1,1) = v.at<float>(0,4);
  outAffMatx.at<float>(2,1) = v.at<float>(3,4);
  outAffMatx.at<float>(0,2) = 0;
  outAffMatx.at<float>(1,2) = 0;
  outAffMatx.at<float>(2,2) = 1;

  //outAffMatx = outAffMatx.t();
  //std::cout << outAffMatx << std::endl << std::endl;
};




void normalizePoints(std::vector<cv::Point2f> &points,
                     std::vector<int> indices,
                     std::vector<cv::Point2f> &samples,
                     cv::Mat & normMat)
{
  int lenVector = points.size();

  std::vector<cv::Point2f> reducedPoints;
  if (indices[0] == -1)
  {
    reducedPoints = points;
  }
  else
  {
    assert(indices.size()>=2);    // Require to use minimum of two points
    reducedPoints = std::vector<cv::Point2f>();
    for (int i=0; i<indices.size(); i++)
    {
      int j = indices[i];
      assert(j<points.size());
      reducedPoints.push_back(points[j]);
    }
  }

  lenVector = reducedPoints.size();

  samples = std::vector<cv::Point2f>();
  float xMean=0,yMean=0;
  for (int i=0; i<lenVector; i++)
  {
    xMean += reducedPoints[i].x;
    yMean += reducedPoints[i].y;
  }
  xMean = xMean/lenVector;
  yMean = yMean/lenVector;

  float std=0;
  for (int i=0; i<lenVector; i++)
  {
    samples.push_back(reducedPoints[i] - cv::Point2f(xMean, yMean));
    std += (samples[i].x * samples[i].x);
    std += (samples[i].y * samples[i].y);
  }

  std = std/(2*lenVector-1);
  std = sqrt(std);

  float weight;
  if (std> 0)
    weight = sqrt(2)/std;
  else
    weight = 1;

  for (int i=0; i<lenVector; i++)
  {
    samples[i] = samples[i] * weight;
  }

  normMat  = cv::Mat(3,3,CV_32F);
  normMat.at<float>(0,0) = 1/weight;
  normMat.at<float>(0,1) = 0;
  normMat.at<float>(0,2) = 0;
  normMat.at<float>(1,0) = 0;
  normMat.at<float>(1,1) = 1/weight;
  normMat.at<float>(1,2) = 0;
  normMat.at<float>(2,0) = xMean;
  normMat.at<float>(2,1) = yMean;
  normMat.at<float>(2,2) = 1;
}

int computeLoopNumberSVD(int sampleSize, int confidence, int pointNum, int inlierNum)
{
  double val1 = log10(1 - (0.01 * (double)confidence));
  double val2 = ((double)inlierNum/(double)pointNum);
  if (sampleSize == 2)
    val2 = val2 * val2;
  else if (sampleSize == 3)
    val2 = val2 * val2 * val2;
  else
    val2 = pow(val2,sampleSize);

  double val3 = log10(1 - val2);
  int out = (int)ceil(val1/val3);
  return out;
};


cv::Mat computeTForm(int sampleSize, 
                  std::vector<cv::Point2f> &points1,
                  std::vector<cv::Point2f> &points2,
                  std::vector<int> indices)
{
  cv::Mat outAffMatx = cv::Mat::eye(3, 3, CV_32F);;
  std::vector<cv::Point2f> samplePoints1, samplePoints2;
  cv::Mat normMat1, normMat2;
  cv::Mat affineMat;

  if (indices.size() == 0 || 
      (indices.size() == 1 && indices[0] != -1))
    goto EXIT;

  normalizePoints(points1, indices, samplePoints1, normMat1);
  normalizePoints(points2, indices, samplePoints2, normMat2);

  
  // First normalize the points
  switch (sampleSize)
  {
  case 2:
    computerSimilaritySVD(samplePoints1, samplePoints2, affineMat);
    break;

  default:
    computerSimilaritySVD(samplePoints1, samplePoints2, affineMat);  // TODO
    break;
  }

  //std::cout << normMat1 << std::endl;
  //std::cout << normMat2 << std::endl;
  //std::cout << affineMat << std::endl;
  //std::cout << affineMat * normMat2 << std::endl;

  cv::solve(normMat1, affineMat * normMat2, outAffMatx,cv::DECOMP_LU);

  //std::cout << (normMat1*outAffMatx - affineMat*normMat2) << std::endl;

  //std::cout << outAffMatx << std::endl;
  outAffMatx = outAffMatx.t();
EXIT:
  return outAffMatx;
}



void transformPointsForward(std::vector<cv::Point2f> &points1,
                     std::vector<cv::Point2f> &points2,
                     cv::Mat affineMat)
{
  //#define CV_32F  5
  //#define CV_64F  6
  int len = points1.size();
  std::vector<cv::Point2f> _points1;
  if (points1 != points2)
  {
    _points1 = points1;
  }
  else
  {
    _points1 = std::vector<cv::Point2f>(len);
    for (int i=0; i<len; i++)
      _points1[i] = points1[i];
  }
  points2.clear();

  if (affineMat.type() == 5)
  {
    for (int i=0; i<_points1.size(); i++)
    {
      cv::Point2f out = cv::Point2f();

      out.x = affineMat.at<float>(0,0) * _points1[i].x
              + affineMat.at<float>(0,1) * _points1[i].y
              + affineMat.at<float>(0,2);
      out.y = affineMat.at<float>(1,0) * _points1[i].x
              + affineMat.at<float>(1,1) * _points1[i].y
              + affineMat.at<float>(1,2);

      points2.push_back(out);
    }
  }
  else if (affineMat.type() == 6)
  {
    for (int i=0; i<_points1.size(); i++)
    {
      cv::Point2f out = cv::Point2f();

      out.x = affineMat.at<double>(0,0) * _points1[i].x
              + affineMat.at<double>(0,1) * _points1[i].y
              + affineMat.at<double>(0,2);
      out.y = affineMat.at<double>(1,0) * _points1[i].x
              + affineMat.at<double>(1,1) * _points1[i].y
              + affineMat.at<double>(1,2);

      points2.push_back(out);
    }
  }
  return;
}

void transformPointsForward(std::vector<cv::Point2i> &points1,
                     std::vector<cv::Point2i> &points2,
                     cv::Mat affineMat)
{
  int len = points1.size();
  std::vector<cv::Point2i> _points1;
  if (points1 != points2)
  {
    _points1 = points1;
  }
  else
  {
    _points1 = std::vector<cv::Point2i>(len);
    for (int i=0; i<len; i++)
      _points1[i] = points1[i];
  }
  points2.clear();

  if (affineMat.type() == 5)
  {
    for (int i=0; i<_points1.size(); i++)
    {
      cv::Point2i out = cv::Point2i();
      out.x = (int)(affineMat.at<float>(0,0) * (float)_points1[i].x
              + affineMat.at<float>(0,1) * (float)_points1[i].y
              + affineMat.at<float>(0,2) + 0.5);
      out.y = (int)(affineMat.at<float>(1,0) * (float)_points1[i].x
              + affineMat.at<float>(1,1) * (float)_points1[i].y
              + affineMat.at<float>(1,2) + 0.5);

      points2.push_back(out);
    }
  }
  else if (affineMat.type() == 5)
  {
    for (int i=0; i<_points1.size(); i++)
    {
      cv::Point2i out = cv::Point2i();
      out.x = (int)(affineMat.at<double>(0,0) * (double)_points1[i].x
              + affineMat.at<double>(0,1) * (double)_points1[i].y
              + affineMat.at<double>(0,2));
      out.y = (int)(affineMat.at<double>(1,0) * (double)_points1[i].x
              + affineMat.at<double>(1,1) * (double)_points1[i].y
              + affineMat.at<double>(1,2));

      points2.push_back(out);
    }
  }

  return;
}
