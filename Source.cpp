
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <iostream>

#include "FST.h"

int main()
{
	char datasetFile[] = "Pathogen_VOC_Dataset_v4.txt";

	int distancevalue=0;
    printf("Enter one of following the distance matrix to be used: \n  0 for Manhattan \n 1 for Euclidean, \n 2 for Chebyshev, \n 3 for Sorensen, \n 4 for SorensenBinary, \n 5 f0r Canberra, \n 6 for Cosine \n  :");
    std::cin >> distancevalue;
    DistanceMetric DistanceMap[] = {
        DistanceMetric::Manhattan,
        DistanceMetric::Euclidean,
        DistanceMetric::Chebyshev,
        DistanceMetric::Sorensen,
        DistanceMetric::SorensenBinary,
        DistanceMetric::Canberra,
        DistanceMetric::Cosine
    };

	int maxK = 12;

	int weightmethod=0;
	printf("Enter 0 for uniform and 1 for weighted method :  ");
	std::cin >>weightmethod;
	bool weighted;
	weightmethod==1?weighted = true:weighted = false;



	FST pathogenTest(datasetFile,DistanceMap[distancevalue],weighted);

	pathogenTest.PrintDataset(false);

	for (int i = 1; i < maxK; i += 2)
		pathogenTest.BackwardFeatureElimination(i, DistanceMap[distancevalue], weighted);


	printf("Press any key to quit...");
	return 0;

}
