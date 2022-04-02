
#include <ctime>
#include <fstream>
#include <string>
#include <thread>
#include "predict.hpp"

// g++ -Wall -O3 .\*.cpp -o .\mainFunc.exe; .\mainFunc.exe

using namespace NeuralNetwork;

void PrintTips(const char *str) {
    std::cout << std::endl;
    for (int i = 0; i < 4; ++i) {
        std::cout << "=== " << str << " ===";
    }
    std::cout << std::endl;
}

void MatrixTest() {
    PrintTips("Matrix Test");
    Matrix::Mat_T<int> arr1(3U, 1U, (const double[3]){1, 2, 3});
    Matrix::Mat_T<float> arr2(4U, 1U, (const double[4]){4, 5, 6, 7});
    Matrix::Mat_T<short> arr3(5U, 1U, (const double[5]){7, 8, 9, 1, 2});
    Matrix::Mat_T<int> mat1 = Matrix::OutMul<int>(arr1, arr2);
    Matrix::Mat_T<int> mat2 = Matrix::OutMul<int>(arr2, arr3);

    mat1.Print(), mat2.Print();
    (mat1 * mat2).Print();

    std::cout << Matrix::DotMul<int>(mat1, arr2) << std::endl;
}

void DerivativeTest(const Image::Img_T &img, std::vector<Para_T<double>> paraSet) {
    PrintTips("NumericDerivative1");
    auto dA1 = CaculateNumGrad(img, paraSet[1].A, paraSet);
    auto dB1 = CaculateNumGrad(img, paraSet[1].B, paraSet);
    dA1.Print(14), std::cout << std::endl;
    dB1.Print(14), std::cout << std::endl;

    PrintTips("NumericDerivative2");
    auto dA2 = CaculateNumGrad(img, paraSet[2].A, paraSet);
    auto dB2 = CaculateNumGrad(img, paraSet[2].B, paraSet);
    dA2.Print(14), std::cout << std::endl;
    dB2.Print(14), std::cout << std::endl;

    CaculateGrad(img, paraSet);
    paraSet[1].A.Print(14), std::cout << std::endl;
    paraSet[1].B.Print(14), std::cout << std::endl;
    paraSet[2].A.Print(14), std::cout << std::endl;
    paraSet[2].B.Print(14), std::cout << std::endl;
}

void ParaReadFromDisk(std::vector<Para_T<double>> &paraSet) {
    paraSet[1].A.ReadFromDisk("./para/A1.dat");
    paraSet[1].B.ReadFromDisk("./para/B1.dat");
    paraSet[2].A.ReadFromDisk("./para/A2.dat");
    paraSet[2].B.ReadFromDisk("./para/B2.dat");
}
void ParaWriteToDisk(const std::vector<Para_T<double>> &paraSet) {
    PrintTips("Save to disk");
    paraSet[1].A.SaveToDisk("./para/A1.dat");
    paraSet[1].B.SaveToDisk("./para/B1.dat");
    paraSet[2].A.SaveToDisk("./para/A2.dat");
    paraSet[2].B.SaveToDisk("./para/B2.dat");
}

void PredictTest() {
    PrintTips("Init image");
    const auto trainImg = Image::GetImgSet("./MNIST/train-images", "./MNIST/train-labels");
    const auto testImg = Image::GetImgSet("./MNIST/t10k-images", "./MNIST/t10k-labels");

    std::cout << "trainImg total numbers: " << trainImg.size() << std::endl;
    std::cout << "trainImg total numbers: " << testImg.size() << std::endl;

    PrintTips("Init parameter");
    uint32_t imgSize = trainImg[0].mat.GetElemNum();

    std::vector<Para_T<double>> paraSet;
    paraSet.push_back(Para_T<double>{});
    paraSet.push_back({100U, imgSize, Matrix::tanh, Matrix::d_tanh});
    paraSet.push_back({10U, 100U, Matrix::softmax, Matrix::d_softmax});

    if (0) {
        DerivativeTest(trainImg[1], paraSet);
    }

    // ParaReadFromDisk(paraSet);

    double accuracy = 0.0, learnRate = 2.0;

    clock_t t0 = clock();

    const uint32_t trainNum = trainImg.size(), trainUnit = 100U;
    for (uint32_t i = 0U; i < trainNum * 10U; i += 100U) {
        const Image::Img_T *img = &trainImg[i % (trainNum - trainUnit)];

        TrainBatchTh<5>(img, trainUnit, paraSet);
        MergeGrad(paraSet, learnRate);

        if (i % 20000U == 20000U - trainUnit) {
            double newAccuracy = GetAccuracy(&trainImg[0], 1000, paraSet);
            if (newAccuracy < accuracy) learnRate *= 0.75;
            accuracy = std::max(accuracy, newAccuracy);

            const double time = (clock() - t0) * 1.0 / CLOCKS_PER_SEC;
            std::cout << "It has trained " << std::setw(6) << i + trainUnit << " images ";
            std::cout << "  time elapse(s):" << std::setw(7) << time;
            std::cout << "  current learnRate " << learnRate << std::endl;
        }
    }

    // PrintTips("Predict");
    // PrintPredict(trainImg[1U], paraSet);
    // PrintPredict(trainImg[123U], paraSet);
    // PrintPredict(testImg[3U], paraSet);
    // PrintPredict(testImg[135U], paraSet);

    PrintTips("GetAccuracy");

    accuracy = GetAccuracy(&trainImg[0], trainImg.size(), paraSet);
    std::cout << "Train accuracy: " << accuracy * 100.0 << std::endl;
    accuracy = GetAccuracy(&testImg[0], testImg.size(), paraSet);
    std::cout << "Test accuracy: " << accuracy * 100.0 << std::endl;

    ParaWriteToDisk(paraSet);

    PrintTips("The end");
}

int main() {
    try {
        if (0) {
            MatrixTest();
        }
        if (1) {
            PredictTest();
        }
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
    }
    return 0;
}
