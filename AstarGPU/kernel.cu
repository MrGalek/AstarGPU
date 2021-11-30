
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>

using namespace std;

struct Edge
{
    double weight;//waga prawdziwa + heurystyczna
    double weightFromStart;//waga od startu
    int idOfNode;//id noda z ktorego ide
    int idOfPreviusNode;//id do ktorego ide

    __device__ __host__ bool operator < (const Edge& e) const
    {
        return weight < e.weight;
    }

};

struct Coord
{
    int x;
    int y;
};

struct Node
{
    int numberOfEdges;
    int* idsOfNeighbor;//tablica z id'kami sasiadow
    double* weightOfEdges;//tablica z wagami krawedzi do sasiadow
};

int getIndexOfNodeByName(vector<string> namesOfNodes, string name);
double calcHeuristicValue(int x1, int y1, int x2, int y2);
double calcHeuristicValue(int x1, int y1);

int main(int argc, char* argv[])
{
    ////DEKLARACJE 

    int numberOfNodes;//licba przypadkow
    vector<string> namesOfNodes;//nazwy - etykiety nodow
    Coord* coords;
    vector<Node> nodes;//wektor z node'ami
    thrust::host_vector<Edge> openEdges;//wekstor z otwartymi akutalnie krawedziami
    thrust::host_vector<Edge> checkedEdges;//sprawdzone krawedzie
    thrust::host_vector<Edge>::iterator itr;//iterator potrzebny do szukania w wektorach

    ////CZYTANIE Z PLIKU

    ifstream inputFile(argv[1]);
    string inputString;

    if (inputFile.is_open())
    {
        inputFile >> inputString;
        numberOfNodes = stoi(inputString);

        coords = new Coord[numberOfNodes];

        for (int i = 0; i < numberOfNodes; i++)
        {
            Node tmpNode;
            inputFile >> inputString;
            namesOfNodes.push_back(inputString);
            inputFile >> inputString;
            coords[i].x = stoi(inputString);
            inputFile >> inputString;
            coords[i].y = stoi(inputString);
            inputFile >> inputString;
            tmpNode.numberOfEdges = stoi(inputString);
            tmpNode.idsOfNeighbor = new int[tmpNode.numberOfEdges];
            tmpNode.weightOfEdges = new double[tmpNode.numberOfEdges];

            for (int j = 0; j < tmpNode.numberOfEdges; j++)
            {
                inputFile >> inputString;
                tmpNode.idsOfNeighbor[j] = stoi(inputString);
                inputFile >> inputString;
                tmpNode.weightOfEdges[j] = stoi(inputString);
            }

            nodes.push_back(tmpNode);
        }
    }
    else
    {
        cout << "File not found. Run program witch parameters: astarInputPoints.txt S G";
        coords = new Coord[1];
        return 0;
    }

    //// ALGORYTM
    
    cout << "START" << endl;

    clock_t beginTime = clock();

    int indexOfFinishNode = getIndexOfNodeByName(namesOfNodes, argv[3]);
    int indexOfStartNode = getIndexOfNodeByName(namesOfNodes, argv[2]);
    int indexOfCurrentNode = indexOfStartNode;
    openEdges.push_back({ 0,0,indexOfStartNode,0 });//dodaje poczatek grafu to otwarych krawedzi

    do
    {
        checkedEdges.push_back(openEdges.front());//dorzucam krawedz do sprawdzonych

        Node currentNode = nodes.at(indexOfCurrentNode);
        itr = thrust::find_if(checkedEdges.begin(), checkedEdges.end(), [&indexOfCurrentNode](Edge e) { return e.idOfNode == indexOfCurrentNode; });
        double weightFromStartOfCurrentNode = itr.base()->weightFromStart;//pobieram rzeczywista wage do aktualnie wybranego node'a

        for (int i = 0; i < currentNode.numberOfEdges; i++)
        {
            Edge tmpEdge;//tworze tymczasowa krawedz
            tmpEdge.idOfPreviusNode = indexOfCurrentNode;//krawedzi prowadzi od akutalnego node'a
            tmpEdge.idOfNode = currentNode.idsOfNeighbor[i];//do sasiada aktualnego node'a
            tmpEdge.weightFromStart = weightFromStartOfCurrentNode + currentNode.weightOfEdges[i];//obliczam rzeczywista wage od startu do sasiada
            tmpEdge.weight = tmpEdge.weightFromStart + calcHeuristicValue(coords[tmpEdge.idOfNode].x, coords[tmpEdge.idOfNode].y, coords[indexOfFinishNode].x, coords[indexOfFinishNode].y);

            itr = thrust::find_if(openEdges.begin(), openEdges.end(), [&tmpEdge](Edge e) {return e.idOfNode == tmpEdge.idOfNode; });

            if (itr != openEdges.end()) //sprawdzam czy ta krawedzi jest juz w wektorze otwartych
            {
                if (itr.base()->weight > tmpEdge.weight)//jezeli jej waga w wektorze jest wieksza to podmieniam ja na tanszy odpowiednik - uzywam iteratora bo to zapewnia turbo 
                {
                    *itr = tmpEdge;
                }
            }
            else //jezeli nie to
            {
                itr = thrust::find_if(checkedEdges.begin(), checkedEdges.end(), [&tmpEdge](Edge e) {return e.idOfNode == tmpEdge.idOfNode; });

                if (itr != checkedEdges.end()) //sprawdzam czy ta krawedzi jest juz w wektorze sprawdzonych
                {
                    if (itr.base()->weight > tmpEdge.weight)//jezeli jej waga w wektorze jest wieksza to podmieniam ja na tanszy odpowiednik - uzywam iteratora bo to zapewnia turbo 
                    {
                        *itr = tmpEdge;
                    }
                }
                else //jezeli nie ma jej w zadnym zbiorze to dorzucam ja do otwartych 
                {
                    openEdges.push_back(tmpEdge);
                }
            }
        }


        openEdges.erase(openEdges.begin());//usuwam sprawdzona krawedz z wektora
        thrust::device_vector<Edge> openEdges_D = openEdges;
        //thrust::sort(openEdges.begin(), openEdges.end());//sortuje krawedzie do odwiedzenia
        thrust::sort(openEdges_D.begin(), openEdges_D.end());//sortuje krawedzie do odwiedzenia
        openEdges = openEdges_D;

        Edge e = openEdges.front();
        indexOfCurrentNode = e.idOfNode;//biore id najtanszej krawedzi
        if (indexOfCurrentNode == indexOfFinishNode)//sprawdzam czy nie jest ona celem 
        {
            checkedEdges.push_back(openEdges.front());//jak tak to dodaje do odwiedzonych
        }

    } while (indexOfCurrentNode != indexOfFinishNode);//wykonuje petle dopoki nie znajdzie celowej krawedzi

    clock_t endTime = clock();
    double elapsed_secs = double(endTime - beginTime) / CLOCKS_PER_SEC;
    cout << "Execution time: " << elapsed_secs << endl;

    Edge currentEdge = checkedEdges.back();//tymczasowa krawedz do doczytania wyniku
    cout << currentEdge.weightFromStart << endl;;

    cout << namesOfNodes.at(indexOfFinishNode) << endl;

    do
    {
        cout << namesOfNodes.at(currentEdge.idOfPreviusNode) << endl;//wyswietam nazwe poprzednika
        int idOfPreviusNode = currentEdge.idOfPreviusNode;
        currentEdge = *thrust::find_if(checkedEdges.begin(), checkedEdges.end(), [&idOfPreviusNode](Edge e) { return e.idOfNode == idOfPreviusNode; }).base();
    } while (currentEdge.idOfNode != indexOfStartNode);//dopoki nie spotkam sie z pierwszym




    ////WYSWIETLANIE WSZYSTKIEGO

    //cout << numberOfNodes << endl;

    //for (int i = 0; i < numberOfNodes; i++)
    //{
    //    cout << nodes.at(i).name << " " << nodes.at(i).coord.x << " " << nodes.at(i).coord.y << " " << nodes.at(i).numberOfEdges;
    //    for (int j = 0; j < nodes.at(i).numberOfEdges; j++)
    //    {
    //        cout << " " << nodes.at(i).idsOfNeighbor[j] << " ";
    //        cout << nodes.at(i).weightOfEdges[j];
    //    }
    //    cout << " " << endl;
    //}

}

int getIndexOfNodeByName(vector<string> namesOfNodes, string name)
{
    auto iterator = find(namesOfNodes.begin(), namesOfNodes.end(), name);

    if (iterator != namesOfNodes.end())
    {
        int index = iterator - namesOfNodes.begin();
        return index;
    }
}

double calcHeuristicValue(int x1, int y1)
{
    return calcHeuristicValue(x1, y1, 0, 0);
}

double calcHeuristicValue(int x1, int y1, int x2, int y2)
{
    return sqrt(pow(((double)x2 - (double)x1), 2) + pow(((double)y2 - (double)y1), 2));
}

//double getWeightFromStartForNode(thrust::host_vector<Edge> edges, int idOfNode)
//{
//    for (int i = 0; i < edges.size(); i++)
//    {
//        Edge e = edges[i];
//        if (e.idOfNode == idOfNode) return e.weightFromStart;
//    }
//
//    return 0;
//}

//bool isEdgeIsOpened(thrust::host_vector<Edge> edges, int idOfNode)
//{
//    for (int i = 0; i < edges.size(); i++)
//    {
//        Edge e = edges[i];
//        if (e.idOfNode == idOfNode) return true;
//    }
//
//    return false;
//}

//int getIndexOfEdge(thrust::host_vector<Edge> edges, int idOfNode)
//{
//    for (int i = 0; i < edges.size(); i++)
//    {
//        Edge e = edges[i];
//        if (e.idOfNode == idOfNode) return i;
//    }
//
//    return 0;
//}