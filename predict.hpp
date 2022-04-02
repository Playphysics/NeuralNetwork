#ifndef PREDICT_HPP_
#define PREDICT_HPP_

#include <iostream>
#include <algorithm>
#include <vector>
#include <thread>
#include "image.h"

namespace NeuralNetwork {

template <typename Elem_T>
struct Para_T final {
    Matrix::Mat_T<Elem_T> A, dA;
    Matrix::Mat_T<Elem_T> B, dB;

    Matrix::Mat_T<Elem_T> (*func)(const Matrix::Mat_T<Elem_T> &);
    Matrix::Mat_T<Elem_T> (*d_func)(const Matrix::Mat_T<Elem_T> &);

    size_t derivativeNum;

    static Elem_T Rand(Elem_T min, Elem_T max) noexcept {
        Elem_T ret = static_cast<Elem_T>(std::rand()) / RAND_MAX;
        return ret * (max - min) + min;
    }

    typedef decltype(func) Func_T;

    Para_T() : A(0), dA(A), B(0), dB(B), func(0), d_func(0) {}
    Para_T(uint32_t outSize, uint32_t inSize, Func_T funcPtr, Func_T d_f);

    Matrix::Mat_T<Elem_T> GetInput(const Matrix::Mat_T<Elem_T> &mat) const {
        if (A.GetElemNum() == 0U || func == 0) {
            return Matrix::Mat_T<Elem_T>(mat);
        }
        return A * mat + B;
    }
    Matrix::Mat_T<Elem_T> GetOutput(const Matrix::Mat_T<Elem_T> &mat) const {
        if (A.GetElemNum() == 0U || func == 0) {
            return Matrix::Mat_T<Elem_T>(mat);
        }
        return func(mat);
    }
    Matrix::Mat_T<Elem_T> operator()(const Matrix::Mat_T<Elem_T> &mat) const {
        if (A.GetElemNum() == 0U || func == 0) {
            return Matrix::Mat_T<Elem_T>(mat);
        }
        return func(A * mat + B);
    }

    void ShowPredict(const Image::Img_T &img) noexcept;
};

// === class method ====== class method ====== class method ====== class method ===

// === class method ====== class method ====== class method ====== class method ===

// === class method ====== class method ====== class method ====== class method ===

template <typename Elem_T>
Para_T<Elem_T>::Para_T(uint32_t outSize, uint32_t inSize, Func_T funcPtr, Func_T d_f)
    : A(outSize, inSize), dA(A), B(outSize), dB(B), func(funcPtr), d_func(d_f) {
    for (uint32_t i = 0U; i < A.GetElemNum(); ++i) {
        A.ptr[i] = Rand(-0.2F, 0.2F);
    }
    for (uint32_t i = 0U; i < B.GetElemNum(); ++i) {
        B.ptr[i] = Rand(-0.0F, 0.0F);
    }
    dA.Clear(), dB.Clear();
    derivativeNum = 0U;
}

template <typename Elem_T>
void Para_T<Elem_T>::ShowPredict(const Image::Img_T &img) noexcept {
    Matrix::Mat_T<Elem_T> predict = operator()(img);
    std::cout << "Predict value: " << predict.MaxLocation << std::endl;
    img.Print();
}

// === function ====== function ====== function ====== function ====== function ===

// === function ====== function ====== function ====== function ====== function ===

// === function ====== function ====== function ====== function ====== function ===

template <typename Elem_T>
Matrix::Mat_T<Elem_T> PredictImg(const Image::Img_T &img,
                                 const std::vector<Para_T<Elem_T>> &paraSet) {
    const size_t paraNum = paraSet.size();
    Matrix::Mat_T<Elem_T> imgMat = img.mat.ConvertType<Elem_T>();

    for (size_t i = 0U; i < paraNum; ++i) {
        imgMat = paraSet[i](imgMat);
    }
    return imgMat;
}

template <typename Elem_T>
Elem_T GetLoss(const Image::Img_T &img, const std::vector<Para_T<Elem_T>> &paraSet) {
    Matrix::Mat_T<Elem_T> pred = PredictImg(img, paraSet);

    pred.ptr[img.label] -= 1;  // predict = predict - real

    return Matrix::DotMul<Elem_T>(pred, pred);
}

template <typename Elem_T>
Matrix::Mat_T<Elem_T> CaculateNumGrad(const Image::Img_T &img, Matrix::Mat_T<Elem_T> &destPara,
                                      const std::vector<Para_T<Elem_T>> &paraSet) {
    constexpr Elem_T h = static_cast<Elem_T>(0.00001) == 0 ? 1 : 0.00001;
    const Elem_T loss = GetLoss(img, paraSet);

    Matrix::Mat_T<Elem_T> ret(destPara.row, destPara.col);

    const uint32_t elemNum = destPara.GetElemNum();
    for (uint32_t i = 0U; i < elemNum; ++i) {
        destPara.ptr[i] += h;
        ret.ptr[i] = (GetLoss(img, paraSet) - loss) / h;
        destPara.ptr[i] -= h;
    }
    return ret;
}

template <typename Elem_T>
void PrintPredict(const Image::Img_T &img, const std::vector<Para_T<Elem_T>> &paraSet) {
    Matrix::Mat_T<Elem_T> pred = PredictImg(img, paraSet);
    std::cout << "Predict value: " << pred.MaxLocation() << std::endl;
    img.Print();
}

template <typename Elem_T>
uint8_t GetPredictNum(const Image::Img_T &img, const std::vector<Para_T<Elem_T>> &paraSet) {
    Matrix::Mat_T<Elem_T> pred = PredictImg(img, paraSet);
    return static_cast<uint8_t>(pred.MaxLocation());
}

template <typename Elem_T>
double GetAccuracy(const Image::Img_T *begin, size_t imgNum,
                   const std::vector<Para_T<Elem_T>> &paraSet) {
    uint32_t correctNum = 0U;
    for (uint32_t i = 0U; i < imgNum; ++i) {
        const uint8_t predNum = GetPredictNum(begin[i], paraSet);
        if (predNum == begin[i].label) correctNum += 1U;
    }
    return correctNum * 1.0 / imgNum;
}

// === train ====== train ====== train ====== train ====== train ====== train ===

// === train ====== train ====== train ====== train ====== train ====== train ===

// === train ====== train ====== train ====== train ====== train ====== train ===

template <typename Elem_T>
void CaculateGrad(const Image::Img_T &img, std::vector<Para_T<Elem_T>> &paraSet) {
    const size_t paraNum = paraSet.size();

    std::vector<Matrix::Mat_T<Elem_T>> in;
    std::vector<Matrix::Mat_T<Elem_T>> out;

    in.push_back(img.mat.ConvertType<Elem_T>());
    out.push_back(in[0]);
    for (size_t i = 1U; i < paraNum; ++i) {
        in.push_back(paraSet[i].GetInput(out[i - 1U]));
        out.push_back(paraSet[i].GetOutput(in[i]));
    }

    Matrix::Mat_T<Elem_T> d_layer = *(out.end() - 1);
    d_layer.ptr[img.label] -= 1, d_layer *= 2;

    for (size_t i = paraNum - 1U; i > 0U; --i) {
        d_layer = paraSet[i].d_func(in[i]) * d_layer;
        paraSet[i].dB += d_layer;
        paraSet[i].dA += Matrix::OutMul<Elem_T>(d_layer, out[i - 1U]);
        paraSet[i].derivativeNum += 1U;

        d_layer = d_layer.Transpose() * paraSet[i].A;
        d_layer.Transpose();
    }
}

template <typename Elem_T>
void MergeGrad(std::vector<Para_T<Elem_T>> &paraSet, Elem_T learnRate = 1) {
    const size_t paraNum = paraSet.size();

    for (size_t i = 0U; i < paraNum; ++i) {
        const Elem_T divNum = paraSet[i].derivativeNum / learnRate;

        paraSet[i].A -= paraSet[i].dA / divNum;
        paraSet[i].B -= paraSet[i].dB / divNum;
        paraSet[i].dA.Clear(), paraSet[i].dB.Clear();
        paraSet[i].derivativeNum = 0U;
    }
}

template <typename Elem_T>
void TrainBatch(const Image::Img_T *begin, const Image::Img_T *end,
                std::vector<Para_T<Elem_T>> *paraSet) {
    for (; begin != end; ++begin) {
        CaculateGrad(*begin, *paraSet);
    }
}

template <uint32_t threadNum, typename Elem_T>
void TrainBatchTh(const Image::Img_T *begin, uint32_t batchSize,
                  std::vector<Para_T<Elem_T>> &paraSet) {
    const auto func = TrainBatch<double>;
    const uint32_t imgPack = batchSize / threadNum;

    std::thread trainThread[threadNum];

    for (uint32_t th = 0U; th < threadNum; ++th) {
        const Image::Img_T *img = begin + imgPack * th;
        trainThread[th] = std::thread(func, img, img + imgPack, &paraSet);
    }
    for (uint32_t th = 0U; th < threadNum; ++th) {
        trainThread[th].join();
    }
}

}  // namespace NeuralNetwork

#endif
