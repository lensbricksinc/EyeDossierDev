#include "blink_detector.h"



int BlinkDetector::addToInternalFaceArray(cv::Rect currFace)
{
    int index = -1;
    for (int i=0; i<5; i++)
    {
        FaceTrackingInfo *faceEntry = &faceArray[i];
        if (faceEntry->isValid)
        {
            int deltaX = (faceEntry->FaceDim.width*4)/5;
            if( abs(faceEntry->FaceDim.x - currFace.x)<deltaX
                && abs(faceEntry->FaceDim.y - currFace.y)<deltaX
                && abs(faceEntry->FaceDim.width - currFace.width)<20)
            {
                index = i;
                break;
            }
        }
    }

    if (index!=-1)
    {
        // Match found
        FaceTrackingInfo *faceEntry = &faceArray[index];

        if (faceEntry->reqValidationCount >0)
            faceEntry->reqValidationCount--;

        faceEntry->FaceDim = currFace;
    }
    else
    {
        // Match not found. Create a new entry.
        for (int j=0; j<5; j++)
        {
            FaceTrackingInfo *faceEntry = &faceArray[j];
            if (!faceEntry->isValid)
            {
                faceEntry->isValid = true; 
                faceEntry->reqValidationCount= FD_VALIDATION_COUNT;
                faceEntry->remPersistenceCount= FD_PERSISTENCE;
                faceEntry->FaceDim = currFace;
                index = j;
                break;
            }
        }

        // Handle the case when all entries are filled up!!!!!!!!!!!!!!!!!!!!!!
    }

    return index;

}



void BlinkDetector::updateExistingFaceArray(int index)
{
    // If i != index, update reqValidationCount, persistanceCount accordingly
    for (int i=0; i<5; i++)
    {
        if (i!= index) {
            FaceTrackingInfo *faceEntry = &faceArray[i];

            if (!faceEntry->isValid)
                continue;

            if (faceEntry->reqValidationCount !=0)
            {
                faceEntry->reqValidationCount = FD_VALIDATION_COUNT;
                faceEntry->reset();
                continue;
            }

            if (faceEntry->remPersistenceCount >0)
                faceEntry->remPersistenceCount --;

            if (faceEntry->remPersistenceCount == 0)
                faceEntry->reset();
        }

    }
    return;
}



cv::Rect BlinkDetector::getBestFaceFrmInternalArray()
{
    int localArray[5];
    memset(localArray, 0, sizeof(int)*5);
    int maxWidthFace = -1;
    int maxWidthFaceIndex = -1;
    int maxPersistenceIndex = -1;
    for (int i=0; i<5; i++)
    {
        FaceTrackingInfo *faceEntry = &faceArray[i];
        if (faceEntry->isValid
            && faceEntry->reqValidationCount == 0
            && faceEntry->remPersistenceCount>0)
        {
            if (maxWidthFace < faceEntry->FaceDim.width)
            {
                maxWidthFace = faceEntry->FaceDim.width;
                maxWidthFaceIndex = i;
            }

            if (faceEntry->remPersistenceCount == FD_PERSISTENCE)
                maxPersistenceIndex = i;
        }

    }

    if (maxPersistenceIndex != -1 
        && (faceArray[maxPersistenceIndex].FaceDim.width > (3*faceArray[maxWidthFaceIndex].FaceDim.width)/4))
        return faceArray[maxPersistenceIndex].FaceDim;
    if (maxWidthFaceIndex != -1)
        return faceArray[maxWidthFaceIndex].FaceDim;
    else
        return cv::Rect(-1,-1,-1,-1);
}


cv::Rect BlinkDetector::postProcessFaces(cv::vector<cv::Rect>& faces)
{
    cv::Rect outFace;
    int index = -1;
    for (int i=0; i<faces.size(); i++)
    {
        // Face validation and persistence
        index = addToInternalFaceArray(faces[i]);
    }
    updateExistingFaceArray(index);
    
    // Display face depending upon internal Face Array
    outFace = getBestFaceFrmInternalArray();


    return outFace;
}