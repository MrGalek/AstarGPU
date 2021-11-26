
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

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

    bool operator < (const Edge& e) const
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
    string name;
    int numberOfEdges;
    int* idsOfNeighbor;//tablica z id'kami sasiadow
    double* weightOfEdges;//tablica z wagami krawedzi do sasiadow
};

int getIndexOfNodeByName(vector<string> namesOfNodes, string name);
double calcHeuristicValue(int x1, int y1, int x2, int y2);
double calcHeuristicValue(int x1, int y1);
double getWeightFromStartForNode(vector<Edge> edges, int idOfNode);
bool isEdgeIsOpened(vector<Edge> edges, int idOfNode);
int getIndexOfEdge(vector<Edge> edges, int idOfNode);

int main(int argc, char* argv[])
{
    ////DEKLARACJE 

    int numberOfNodes;//licba przypadkow
    vector<string> namesOfNodes;//nazwy - etykiety nodow
    Coord* coords;
    vector<Node> nodes;//wektor z node'ami
    vector<Edge> openEdges;//wekstor z otwartymi akutalnie krawedziami
    vector<Edge> checkedEdges;//sprawdzone krawedzie

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
            tmpNode.name = inputString;
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

    int indexOfFinishNode = getIndexOfNodeByName(namesOfNodes, argv[3]);
    int indexOfStartNode = getIndexOfNodeByName(namesOfNodes, argv[2]);
    int indexOfCurrentNode = indexOfStartNode;
    openEdges.push_back({ 0,0,indexOfStartNode,0 });//dodaje poczatek grafu to otwarych krawedzi

    do
    {
        checkedEdges.push_back(openEdges.front());//dorzucam krawedz do sprawdzonych

        Node currentNode = nodes.at(indexOfCurrentNode);
        double weightFromStartOfCurrentNode = getWeightFromStartForNode(checkedEdges, indexOfCurrentNode); //pobieram rzeczywista wage do aktualnie wybranego node'a

        for (int i = 0; i < currentNode.numberOfEdges; i++)
        {
            Edge tmpEdge;//tworze tymczasowa krawedz
            tmpEdge.idOfPreviusNode = indexOfCurrentNode;//krawedzi prowadzi od akutalnego node'a
            tmpEdge.idOfNode = currentNode.idsOfNeighbor[i];//do sasiada aktualnego node'a
            tmpEdge.weightFromStart = weightFromStartOfCurrentNode + currentNode.weightOfEdges[i];//obliczam rzeczywista wage od startu do sasiada
            tmpEdge.weight = tmpEdge.weightFromStart + calcHeuristicValue(coords[tmpEdge.idOfNode].x, coords[tmpEdge.idOfNode].y, coords[indexOfFinishNode].x, coords[indexOfFinishNode].y);

            if (isEdgeIsOpened(openEdges, tmpEdge.idOfNode)) //sprawdzam czy ta krawedzi jest juz w wektorze otwartych
            {
                int iterator = getIndexOfEdge(openEdges, tmpEdge.idOfNode);//jak tak to sprawdzam gdzie jest

                if (openEdges.at(iterator).weight > tmpEdge.weight)//jezeli jej waga w wektorze jest wieksza to podmieniam ja na tanszy odpowiednik
                {
                    openEdges.at(iterator) = tmpEdge;
                }
            }
            else if (isEdgeIsOpened(checkedEdges, tmpEdge.idOfNode)) //sprawdzam czy ta krawedzi jest juz w wektorze sprawdzonych
            {
                int iterator = getIndexOfEdge(checkedEdges, tmpEdge.idOfNode);//jak tak to sprawdzam gdzie jest

                if (checkedEdges.at(iterator).weight > tmpEdge.weight)//jezeli jej waga w wektorze jest wieksza to podmieniam ja na tanszy odpowiednik
                {
                    checkedEdges.at(iterator) = tmpEdge;
                }
            }
            else
            {
                openEdges.push_back(tmpEdge);
            }
        }


        openEdges.erase(openEdges.begin());//usuwam sprawdzona krawedz z wektora
        sort(openEdges.begin(), openEdges.end());//sortuje krawedzie do odwiedzenia

        indexOfCurrentNode = openEdges.front().idOfNode;//biore id najtanszej krawedzi
        if (indexOfCurrentNode == indexOfFinishNode)//sprawdzam czy nie jest ona celem 
        {
            checkedEdges.push_back(openEdges.front());//jak tak to dodaje do odwiedzonych
        }

    } while (indexOfCurrentNode != indexOfFinishNode);//wykonuje petle dopoki nie znajdzie celowej krawedzi

    Edge currentEdge = checkedEdges.back();//tymczasowa krawedz do doczytania wyniku
    cout << currentEdge.weightFromStart << endl;;

    cout << namesOfNodes.at(indexOfFinishNode) << endl;

    do
    {
        cout << namesOfNodes.at(currentEdge.idOfPreviusNode) << endl;//wyswietam nazwe poprzednika
        int idOfPreviusNode = currentEdge.idOfPreviusNode;
        currentEdge = checkedEdges.at(getIndexOfEdge(checkedEdges, idOfPreviusNode));
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

double getWeightFromStartForNode(vector<Edge> edges, int idOfNode)
{
    for (int i = 0; i < edges.size(); i++)
    {
        if (edges.at(i).idOfNode == idOfNode) return edges.at(i).weightFromStart;
    }

    return 0;
}

bool isEdgeIsOpened(vector<Edge> edges, int idOfNode)
{
    for (int i = 0; i < edges.size(); i++)
    {
        if (edges.at(i).idOfNode == idOfNode) return true;
    }

    return false;
}

int getIndexOfEdge(vector<Edge> edges, int idOfNode)
{
    for (int i = 0; i < edges.size(); i++)
    {
        if (edges.at(i).idOfNode == idOfNode) return i;
    }

    return 0;
}