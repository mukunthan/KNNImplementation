#pragma once

#ifndef FST_H
#define FST_H

#include <vector>
#include <string>

using namespace std;

enum DistanceMetric{Manhattan, Euclidean, Chebyshev, Sorensen, SorensenBinary, Canberra, Cosine};

class FST //Feature Selection Tools
{
private:

	int noClasses;			//number of different classes
	int noSamples;			//total number of samples in the dataset
	int noFeatures;			//number of dimensions of each feature vector
	int maxFeatureLen;		//maximum feature length for selected features
	double OverallBestResultforallK=0.0;

	//int **data;				//dataset matrix
	vector<vector<int>>	data;

	//int *classIDs;			//class ID of ith vector in the data matrix
	vector<int>	classIDs;

	//char **classLabels;		//class labels
	vector<string>	classLabels;

	//char dataFilename[500];	//dataset file path
	string dataFilename, resultsFilename;


protected:

	inline double DistEuclidean(vector<int> v1, int v2index, vector<int> fVecIDs);

	inline double DistManhattan(vector<int> v1, int v2, vector<int> fVecIDs);

	inline double DistChebyshev(vector<int> v1, vector<int> v2, vector<int> fVecIDs);

	inline double DistSorensen(vector<int> v1, int v2, vector<int> fVecIDs);

	inline double DistSorensenBinary(vector<int> v1, int v2, vector<int> fVecIDs);

	inline double DistCanberra(vector<int> v1, int v2, vector<int> fVecIDs);

	inline double SimiCosine(vector<int> v1, int v2, vector<int> fVecIDs);

	inline double ComputeDistance(vector<int> v1, int v2, vector<int> fVecIDs, DistanceMetric metric);

public:

	FST(const char* datasetfile,DistanceMetric distancemetric,bool weighted);

	FST(int noClasses, int noSamples, int noDims, int maxFeatLen);

	~FST();

	//Sequential Feature Selection
	void SequentialFeatureSelection(int K, DistanceMetric metric, bool distWeight);

	//Backward Feature Elimination
	void BackwardFeatureElimination(int K, DistanceMetric metric, bool distWeight);

	void LeaveOneOutClassification(int K, DistanceMetric metric, bool distWeight);

	void ReadDataset();

	void PrintDataset(bool withVectors);

	void PrintDistMetric(DistanceMetric metric);

	void PrintDistMetricToFile(DistanceMetric metric, FILE *file);

};




#endif
