#include <stdio.h>
#include <string>
#include <math.h>
#include <ctime>

#include <vector>
#include <tuple>
#include <algorithm> //<bits/stdc++.h>


#include "FST.h"


FST::FST(const char* datasetfile,DistanceMetric distancemetric,bool weighted)
{
	//strcpy(dataFilename, datasetfile);
	dataFilename = datasetfile;

	ReadDataset();

	maxFeatureLen = 50;

	resultsFilename = "Results_"+std::to_string(distancemetric)+std::to_string(weighted)+".txt";
	FILE *outFile = fopen(resultsFilename.c_str(), "w");	//reset file
	fclose(outFile);
}

FST::FST(int _noClasses, int _noSamples, int _noFeatures, int _maxFeatLen)
{
	noClasses = _noClasses;
	noSamples = _noSamples;
	noFeatures = _noFeatures;
	maxFeatureLen = _maxFeatLen;
}

FST::~FST()
{

}

void FST::SequentialFeatureSelection(int K, DistanceMetric metric, bool distWeight)
{
	double overallBestAcc = 0;
	time_t curr_time = time(NULL);
	char *tm = ctime(&curr_time);

	printf("\n____________________________________________________________________________");
	printf("\nSequential Forward Feature Selection has started with metric: ");
	PrintDistMetric(metric);
	printf(" and K=%d on (%s\n", K, tm);

	FILE *outFile = fopen(resultsFilename.c_str(), "a+");
	fprintf(outFile, "\n_______________________________________________________________");
	fprintf(outFile, "\nSequential Forward Feature Selection has started with metric : ");
	PrintDistMetricToFile(metric, outFile);
	fprintf(outFile, " and K=%d on (%s\n", K, tm);

	for (int i = 0; i < noFeatures; i++)
	{
		clock_t begin = clock();
		double bestAcc = 0;
		vector<int> selFeatIDs;
		selFeatIDs.reserve(maxFeatureLen);
		selFeatIDs.push_back(i);	//ith feature is automatically added, always starts with 2 features

		for (int j, jj = 0; jj < noFeatures; jj++)
		{
			j = (i + 1 + jj) % noFeatures;	//circular iteration
			selFeatIDs.push_back(j);		//jth feature is appended

			//Leave-1-Out Classification
			double TP = 0;	//# true positives
			for (int m = 0; m < noSamples; m++)
			{
				vector<int> curFeatVec;		//prepare current feature vector
				for (int p = 0; p < selFeatIDs.size(); p++)
					curFeatVec.push_back(data[m][selFeatIDs[p]]);

				vector<int> tClassIDs;
				vector<double> distances;
				for (int n = 0; n < noSamples; n++)	//Compute all distances
				{
					if (m == n) continue;

					double dist = ComputeDistance(curFeatVec, n, selFeatIDs, metric);

					distances.push_back(dist);
					tClassIDs.push_back(classIDs[n]);
				}

				vector<int> nearestClassIDs;
				vector<double> nearestDistances;
				for (int p = 0; p < K; p++)		//Find K mininmum distances
				{
					int index;
					double minDist = 1e5;	//a large number
					for (int q = 0; q < distances.size(); q++)
					{
						if (distances[q] < minDist)
						{
							minDist = distances[q];
							index = q;
						}
					}
					nearestClassIDs.push_back(tClassIDs[index]);
					nearestDistances.push_back(minDist);	//to be used in weughted dist calculations
					distances[index] = 1e5;					//prevent getting the same distance again
				}

				int* results = new int[noClasses + 1]{ 0 };	//k-histogram
				if (distWeight)
					for (int p = 0; p < nearestClassIDs.size(); p++)
						if (nearestDistances[p] == 0)
							results[nearestClassIDs[p]] += 10e10;
						else
							results[nearestClassIDs[p]] += 1 / nearestDistances[p];
				else
					for (int p = 0; p < nearestClassIDs.size(); p++)
						results[nearestClassIDs[p]]++;

				double max = 0;
				int classRes = 0;
				for (int p = 1; p <= noClasses; p++) {
					if (results[p] > max) {
						max = results[p];
						classRes = p;
					}
				}
				delete results;

				if (classRes == classIDs[m]) TP++;
			}
			double curAcc = TP / noSamples;	//leave-1-out acc

			if (curAcc > bestAcc)
			{
				bestAcc = curAcc;
				if (bestAcc > overallBestAcc) overallBestAcc = bestAcc;
				//printf("(i: %d Fsize: %d TP: %d) ", j, selFeatIDs.size(), (int)TP);
			}
			else
				selFeatIDs.pop_back();

			if (selFeatIDs.size() > maxFeatureLen) break;
		}
		clock_t end = clock();
		printf("%d / %d is done in %4.1f seconds. K = %d. No. Selected Features: %d \n", i + 1, noFeatures, double(end - begin) / CLOCKS_PER_SEC, K, selFeatIDs.size());
		printf("##### Best Accuracy: %5.3f ##### Overall Best Accuracy: %5.3f #####\nSelected Feature IDs: ", bestAcc, overallBestAcc);
		for (int j = 0; j < selFeatIDs.size(); j++) printf("%d, ", selFeatIDs[j]);
		printf("\b \n\n");
	}
	printf("\nSequential Forward Feature Selection has COMPLETED\n");
}

void FST::BackwardFeatureElimination(int K, DistanceMetric metric, bool distWeight)
{
	double overallBestAcc = 0;
	time_t curr_time = time(NULL);
	char *tm = ctime(&curr_time);

	printf("\n____________________________________________________________________________");
	printf("\nBackward Feature Elimination has started with metric: ");
	PrintDistMetric(metric);
	printf(" and K=%d and weighted=%d, on (%s\n", K,distWeight, tm);

	FILE *outFile = fopen(resultsFilename.c_str(), "a+");	//reset file
	fprintf(outFile, "\n____________________________________________________________________________");
	fprintf(outFile, "\nBackward Feature Elimination Selection has started with metric : ");
	PrintDistMetricToFile(metric, outFile);
	fprintf(outFile, " and K=%d and weighted=%d,on (%s\n", K,distWeight, tm);
	fclose(outFile);

	for (int i = 0; i < noFeatures; i++)
	{
		clock_t begin = clock();
		double bestAcc = 0;
		vector<int> selFeatIDs;
		//select all features in a circular manner
		for (int j = 0; j < noFeatures; j++) selFeatIDs.push_back((i + j) % noFeatures);

		for (int j = noFeatures; j >= 0 ; j--)
		{
			int eliminated = 0;
			if (j < noFeatures)	//no elimination in the first loop
			{
				eliminated = selFeatIDs[j];
				selFeatIDs.erase(selFeatIDs.begin() + j);	//jth feature is appended
			}

			//Leave-1-Out Classification
			double TP = 0;	//# true positives
			for (int m = 0; m < noSamples; m++)
			{
				vector<int> curFeatVec;		//prepare current feature vector
				for (int p = 0; p < selFeatIDs.size(); p++)
					curFeatVec.push_back(data[m][selFeatIDs[p]]);

				vector<int> tClassIDs;
				vector<double> distances;
				for (int n = 0; n < noSamples; n++)	//Compute all distances
				{
					if (m == n) continue;

					double dist = ComputeDistance(curFeatVec, n, selFeatIDs, metric);

					distances.push_back(dist);
					tClassIDs.push_back(classIDs[n]);
				}

				vector<int> nearestClassIDs;
				vector<double> nearestDistances;
				for (int p = 0; p < K; p++)		//Find K mininmum distances
				{
					int index;
					double minDist = 10e10;	//a large number
					for (int q = 0; q < distances.size(); q++)
					{
						if (distances[q] < minDist)
						{
							minDist = distances[q];
							index = q;
						}
					}
					nearestClassIDs.push_back(tClassIDs[index]);
					nearestDistances.push_back(minDist);	//to be used in weighted dist calculations
					distances[index] = 1e5;					//prevent getting the same distance again
				}

				double* results = new double[noClasses + 1]{ 0 };	//k-histogram
				if (distWeight)
					for (int p = 0; p < nearestClassIDs.size(); p++)
						if (nearestDistances[p] == 0)
							results[nearestClassIDs[p]] += 10e10;	//a very large number
						else
							results[nearestClassIDs[p]] += 1 / nearestDistances[p];
				else
					for (int p = 0; p < nearestClassIDs.size(); p++)
						results[nearestClassIDs[p]]++;

				double max = 0;
				int classRes = 0;
				for (int p = 1; p <= noClasses; p++) {
					if (results[p] > max) {
						max = results[p];
						classRes = p;
					}
				}
				delete results;

				if (classRes == classIDs[m]) TP++;
			}
			double curAcc = TP / noSamples;	//leave-1-out acc

			if (curAcc >= bestAcc)
			{
				bestAcc = curAcc;
				if (bestAcc > overallBestAcc)
				{
					overallBestAcc = bestAcc;
					if(overallBestAcc > OverallBestResultforallK) OverallBestResultforallK = overallBestAcc;
				}
				//printf("(i: %d Fsize: %d TP: %d) ", j, selFeatIDs.size(), (int)TP);
			}
			else
				selFeatIDs.insert(selFeatIDs.begin() + j, eliminated);

			if (selFeatIDs.size() < 3) break;
		}
		clock_t end = clock();
		printf("%d / %d is done in %4.1f seconds. K = %d. No. Selected Features: %d \n", i + 1, noFeatures, double(end - begin) / CLOCKS_PER_SEC, K, selFeatIDs.size());
		printf("##### Best Accuracy: %5.3f ##### Overall Best Accuracy: %5.3f   TheBest: %5.3f #####\nSelected Feature IDs:", bestAcc, overallBestAcc,OverallBestResultforallK);
		for (int j = 0; j < selFeatIDs.size(); j++) printf("%d, ", selFeatIDs[j]);
		printf("\b \n\n");

		outFile = fopen(resultsFilename.c_str(), "a+");	//reset file
		fprintf(outFile, "%d / %d is done in %4.1f seconds. K = %d. No. Selected Features: %d \n", i + 1, noFeatures, double(end - begin) / CLOCKS_PER_SEC, K, selFeatIDs.size());
		fprintf(outFile, "##### Best Accuracy: %5.3f ##### Overall Best Accuracy: %5.3f TheBest: %5.3f #####\nSelected Feature IDs:", bestAcc, overallBestAcc,OverallBestResultforallK);
		for (int j = 0; j < selFeatIDs.size(); j++) fprintf(outFile, "%d, ", selFeatIDs[j]);
		fprintf(outFile, "\n\n");
		fclose(outFile);
	}
	
	printf("\Backward Feature Elimination has COMPLETED\n");
}

void FST::LeaveOneOutClassification(int K, DistanceMetric metric, bool distWeight)
{
	time_t curr_time = time(NULL);
	char *tm = ctime(&curr_time);

	printf("Leave-One-Out Classification has with metric: ");
	PrintDistMetric(metric);
	printf(" and K=%d ", K);

	vector<int> selFeatIDs;
	clock_t begin = clock();
	for (int i = 0; i < noFeatures; i++) selFeatIDs.push_back(i);

	double TP = 0;	//# true positives
	for (int m = 0; m < noSamples; m++)
	{
		vector<int> tClassIDs;
		vector<double> distances;
		for (int n = 0; n < noSamples; n++)	//Compute all distances
		{
			if (m == n) continue;

			double dist = ComputeDistance(data[m], n, selFeatIDs, metric);

			distances.push_back(dist);
			tClassIDs.push_back(classIDs[n]);
		}

		vector<int> nearestClassIDs;
		vector<double> nearestDistances;
		for (int p = 0; p < K; p++)		//Find K mininmum distances
		{
			int index;
			double minDist = 1e5;	//a large number
			for (int q = 0; q < distances.size(); q++)
			{
				if (distances[q] < minDist)
				{
					minDist = distances[q];
					index = q;
				}
			}
			nearestClassIDs.push_back(tClassIDs[index]);
			nearestDistances.push_back(minDist);	//to be used in weughted dist calculations
			distances[index] = 1e5;					//prevent getting the same distance again
		}

		int* results = new int[noClasses + 1]{ 0 };	//k-histogram
		if (distWeight)
			for (int p = 0; p < nearestClassIDs.size(); p++)
				if (nearestDistances[p] == 0)
					results[nearestClassIDs[p]] += 10e10;
				else
					results[nearestClassIDs[p]] += 1 / nearestDistances[p];
		else
			for (int p = 0; p < nearestClassIDs.size(); p++)
				results[nearestClassIDs[p]]++;

		double max = 0;
		int classRes = 0;
		for (int p = 1; p <= noClasses; p++) {
			if (results[p] > max) {
				max = results[p];
				classRes = p;
			}
		}
		delete results;

		if (classRes == classIDs[m]) TP++;
	}
	double accuracy = TP / noSamples;	//leave-1-out acc

	clock_t end = clock();
	printf("has completed in %4.1f seconds.\n", double(end - begin) / CLOCKS_PER_SEC);
	printf("##### Accuracy: %5.3f ##### \n\n", accuracy);
}

inline double FST::ComputeDistance(vector<int> v1, int v2, vector<int> fVecIDs, DistanceMetric metric)
{
	double dist;

	switch (metric)
	{
	case DistanceMetric::Manhattan:
		dist = DistManhattan(v1, v2, fVecIDs); break;

	case DistanceMetric::Euclidean:
		dist = DistEuclidean(v1, v2, fVecIDs); break;

	case DistanceMetric::Chebyshev:		break;

	case DistanceMetric::Sorensen:
		dist = DistSorensen(v1, v2, fVecIDs); break;

	case DistanceMetric::SorensenBinary:
		dist = DistSorensenBinary(v1, v2, fVecIDs); break;

	case DistanceMetric::Canberra:
		dist = DistCanberra(v1, v2, fVecIDs); break;

	case DistanceMetric::Cosine:
		dist = SimiCosine(v1, v2, fVecIDs); break;
	}

	return dist;
}

inline double FST::DistEuclidean(vector<int> v1, int v2index, vector<int> fVecIDs)
{
	double tmp, sum = 0;
	//for (int i = 0; i < v1.size(); i++)
	//	sum += pow(v1[i] - data[v2index][fVecIDs[i]], 2);

	for (int i = 0; i < v1.size(); i++)
	{
		//sum += pow(v1[i] - v2[fVecIDs[i]], 2);
		tmp = (v1[i] - data[v2index][fVecIDs[i]]);
		sum += tmp * tmp;
	}

	return sqrt(sum);
}

inline double FST::DistManhattan(vector<int> v1, int v2, vector<int> fVecIDs)
{
	double sum = 0;
	for (int i = 0; i < v1.size(); i++)
		sum += abs(v1[i] - data[v2][fVecIDs[i]]);

	return sum;
}

inline double FST::DistChebyshev(vector<int> v1, vector<int> v2, vector<int> fVecIDs)
{
	double num = 0, denom = 0, dist = 0;
	for (int i = 0; i < v1.size(); i++)
	{
		//to be completed...
	}

	return dist;
}

inline double FST::DistSorensen(vector<int> v1, int v2, vector<int> fVecIDs)
{
	double sumNum = 0, sumDenom = 0;
	for (int i = 0; i < v1.size(); i++)
	{
		sumNum += abs(v1[i] - data[v2][fVecIDs[i]]);
		sumDenom += v1[i] + data[v2][fVecIDs[i]];
	}

	return sumNum / sumDenom;
}

inline double FST::DistSorensenBinary(vector<int> v1, int v2, vector<int> fVecIDs)
{
	double intersection = 0, v1Cardinality = 0, v2Cardinality = 0;
	for (int i = 0; i < v1.size(); i++)
	{
		if (v1[i] == data[v2][fVecIDs[i]] && v1[i] != 0) intersection++;
		if (v1[i] != 0) v1Cardinality++;
		if (data[v2][fVecIDs[i] != 0]) v2Cardinality++;
	}

	return (1- (2 * intersection / (v1Cardinality + v2Cardinality)));
}

inline double FST::DistCanberra(vector<int> v1, int v2, vector<int> fVecIDs)
{
	double num = 0, denom = 0, dist = 0;
	for (int i = 0; i < v1.size(); i++)
	{
		num = abs(v1[i] - data[v2][fVecIDs[i]]);
		denom = abs(v1[i]) + abs(data[v2][fVecIDs[i]]);

		if (denom == 0) denom = 10e-10;	//to void div by zero
		dist += num / denom;
	}

	return dist;
}

inline double FST::SimiCosine(vector<int> v1, int v2, vector<int> fVecIDs)
{
	double sumXY = 0, sumXX = 0, sumYY = 0;
	for (int i = 0; i < v1.size(); i++)
	{
		sumXY += v1[i] * data[v2][fVecIDs[i]];
		sumXX += v1[i] * v1[i];
		sumYY += data[v2][fVecIDs[i]] * data[v2][fVecIDs[i]];
	}
	double cosSimilarity = (sumXX * sumYY) ? sumXY / (sumXX * sumYY) : 10e-5; //if denom is zero, returns a very small number
	return 1 / cosSimilarity;	//distance = 1 / similarity
}

void FST::ReadDataset()
{
	//open & read data file
	FILE *dataFile = fopen(dataFilename.c_str(), "r");

	int tmpInt1, tmpInt2;
	string tmpStr;
	char* buffer = new char[200];

	fscanf(dataFile, "%d", &noClasses);	//# of classes
	fgets(buffer, 100, dataFile);		//dummy read till read EOL


	classLabels.push_back("");	//dummy push to start index from 1 for class labels
	for (int i = 1; i <= noClasses; i++)
	{
		fscanf(dataFile, "%d  %s", &tmpInt1, buffer);
		classLabels.push_back(string(buffer));
	}

	fscanf(dataFile, "%d", &noSamples);		//reads # of classes from the file
	fgets(buffer, 100, dataFile);			//dummy read till read EOL

	fscanf(dataFile, "%d", &noFeatures);	//reads the feature dimension from the file
	fgets(buffer, 100, dataFile);			//dummy read till read EOL

	for (int i = 0; i < noSamples; i++)
	{
		fscanf(dataFile, "%d %d", &tmpInt1, &tmpInt2);	//read sample ID (dummy read) & class ID of the current feature vector
		classIDs.push_back(tmpInt2);	//store class ID

		data.push_back(vector<int>());
		for (int j = 0; j < noFeatures; j++)
		{
			fscanf(dataFile, "%d", &tmpInt1);	//read feature vector
			data[i].push_back(tmpInt1);
		}
	}

	delete buffer;
}

void FST::PrintDataset(bool withVectors)
{
	printf("DATASET INFO \nNo. Classes:\t%3d \nNo. Samples:\t%3d\nFeature Dimen.: %3d\n", noClasses, noSamples, noFeatures);

	printf("\nClass IDs and Labels:\n");
	for (int i = 1; i <= noClasses; i++)
		printf("%4d \t %s\n", i, classLabels[i].c_str());
	printf("\n");
}

void FST::PrintDistMetric(DistanceMetric metric)
{
	switch (metric)
	{
	case DistanceMetric::Manhattan:	printf("Manhattan"); break;
	case DistanceMetric::Euclidean: printf("Euclidean"); break;
	case DistanceMetric::Chebyshev: printf("Chebyshev"); break;
	case DistanceMetric::Sorensen: 	printf("Sorensen"); break;
	case DistanceMetric::SorensenBinary: printf("SorensenBinary"); break;
	case DistanceMetric::Canberra: printf("Canberra"); break;
	case DistanceMetric::Cosine: printf("Cosine"); break;
	}
}

void FST::PrintDistMetricToFile(DistanceMetric metric, FILE *file)
{
	switch (metric)
	{
	case DistanceMetric::Manhattan:	fprintf(file, "Manhattan"); break;
	case DistanceMetric::Euclidean: fprintf(file, "Euclidean"); break;
	case DistanceMetric::Chebyshev: fprintf(file, "Chebyshev"); break;
	case DistanceMetric::Sorensen: 	fprintf(file, "Sorensen"); break;
	case DistanceMetric::SorensenBinary: fprintf(file, "SorensenBinary"); break;
	case DistanceMetric::Canberra: fprintf(file, "Canberra"); break;
	case DistanceMetric::Cosine: fprintf(file, "Cosine"); break;
	}
}
