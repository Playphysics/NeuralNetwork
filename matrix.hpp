#ifndef MATRIX_HPP_
#define MATRIX_HPP_

#include <iostream>
#include <iomanip>
#include <algorithm>
#include <fstream>
#include <cmath>
#include <cassert>

namespace Matrix {

inline void PrintMemInfo(const void *ptr, const char *msg) {
    if (1) {
        std::cout << ptr << ": " << msg << std::endl;
    }
}

template <typename Elem_T>
struct Mat_T final {
    typedef Elem_T MatElem_T;

    uint32_t row, col;
    Elem_T *ptr = nullptr;

    inline uint32_t GetElemNum() const noexcept {
        return row * col;
    }
    inline int GetDim() const noexcept {
        return static_cast<int>(row > 1U) + static_cast<int>(col > 1U);
    }

    template <typename T2 = Elem_T>
    Mat_T(uint32_t matRow = 0U, uint32_t matCol = 1U, const T2 *initData = nullptr);
    Mat_T(const Mat_T &mat) : Mat_T(mat.row, mat.col, mat.ptr) {}
    Mat_T(Mat_T &&mat) noexcept {
        row = mat.row, col = mat.col, ptr = mat.ptr;
        mat.ptr = nullptr;
    }
    Mat_T &operator=(Mat_T mat) noexcept;
    Mat_T &Clear(Elem_T newValue = Elem_T{}) noexcept;
    template <typename New_T>
    Mat_T<New_T> ConvertType() const;
    Mat_T &Transpose();

    Elem_T Min() const noexcept;
    Elem_T Max() const noexcept;
    uint32_t MinLocation() const noexcept;
    uint32_t MaxLocation() const noexcept;

    Mat_T &ReadFromDisk(const char *path);
    void SaveToDisk(const char *path) const;

    void PrintLine(uint32_t row, int width = 8, std::ostream &outStream = std::cout) const;
    void Print(int width = 8, std::ostream &outStream = std::cout) const;
    void PrintDim() const {
        std::cout << this << " dimension: [" << row << ", " << col << ']' << std::endl;
    }

    ~Mat_T() {
        if (ptr) LogMemChange("Destruct");
        delete[] ptr;
    }

   private:
    void LogMemChange(const char *msg) {
        // std::cout << this << ": " << msg << std::endl;
    }
};

// === class method ====== class method ====== class method ====== class method ===

// === class method ====== class method ====== class method ====== class method ===

// === class method ====== class method ====== class method ====== class method ===

template <typename Elem_T>
template <typename T2>
Mat_T<Elem_T>::Mat_T(uint32_t matRow, uint32_t matCol, const T2 *initData)
    : row(matRow), col(matCol) {
    const uint32_t elemNum = this->GetElemNum();
    if (elemNum != 0U) {
        LogMemChange("Construct from value or copy");
        this->ptr = new Elem_T[elemNum];
        if (initData != nullptr) {
            std::copy_n(initData, elemNum, this->ptr);
        }
    }
}

template <typename Elem_T>
Mat_T<Elem_T> &Mat_T<Elem_T>::operator=(Mat_T<Elem_T> mat) noexcept {
    std::swap(this->row, mat.row);
    std::swap(this->col, mat.col);
    std::swap(this->ptr, mat.ptr);
    return *this;
}

template <typename Elem_T>
Mat_T<Elem_T> &Mat_T<Elem_T>::Clear(Elem_T newValue) noexcept {
    std::fill_n(ptr, GetElemNum(), newValue);
    return *this;
}

template <typename Elem_T>
template <typename New_T>
Mat_T<New_T> Mat_T<Elem_T>::ConvertType() const {
    Mat_T<New_T> ret(this->row, this->col);
    std::copy_n(this->ptr, this->GetElemNum(), ret.ptr);
    return ret;
}

template <typename Elem_T>
Mat_T<Elem_T> &Mat_T<Elem_T>::Transpose() {
    if (this->row == 1U || this->col == 1U) {
        std::swap(this->row, this->col);
    } else {
        Mat_T<Elem_T> ret(this->row, this->col);
        for (uint32_t i = 0U; i < row; ++i) {
            for (uint32_t j = 0U; j < col; ++j) {
                std::swap(this->ptr[i * col + j], ret.ptr[j * row + i]);
            }
        }
        *this = ret;
    }
    return *this;
}

template <typename Elem_T>
Elem_T Mat_T<Elem_T>::Min() const noexcept {
    uint32_t elemNum = this->GetElemNum();
    Elem_T dest = ptr[0];

    for (uint32_t i = 1U; i < elemNum; ++i) {
        if (ptr[i] < dest) dest = ptr[i];
    }
    return dest;
}
template <typename Elem_T>
Elem_T Mat_T<Elem_T>::Max() const noexcept {
    uint32_t elemNum = this->GetElemNum();
    Elem_T dest = ptr[0];

    for (uint32_t i = 1U; i < elemNum; ++i) {
        if (ptr[i] > dest) dest = ptr[i];
    }
    return dest;
}
template <typename Elem_T>
uint32_t Mat_T<Elem_T>::MinLocation() const noexcept {
    uint32_t ret = 0U, elemNum = this->GetElemNum();
    Elem_T dest = ptr[ret];

    for (uint32_t i = 0U; i < elemNum; ++i) {
        if (ptr[i] < dest) {
            dest = ptr[i], ret = i;
        }
    }
    return ret;
}
template <typename Elem_T>
uint32_t Mat_T<Elem_T>::MaxLocation() const noexcept {
    uint32_t ret = 0U, elemNum = this->GetElemNum();
    Elem_T dest = ptr[ret];

    for (uint32_t i = 0U; i < elemNum; ++i) {
        if (ptr[i] > dest) {
            dest = ptr[i], ret = i;
        }
    }
    return ret;
}

template <typename Elem_T>
Mat_T<Elem_T> &Mat_T<Elem_T>::ReadFromDisk(const char *path) {
    const auto openMode = std::ios_base::binary | std::ios_base::in;
    std::ifstream file(path, openMode);
    uint32_t srcRow = 0U, srcCol = 0U;

    file.read(reinterpret_cast<char *>(&srcRow), sizeof(srcRow));
    file.read(reinterpret_cast<char *>(&srcCol), sizeof(srcCol));

    Mat_T<Elem_T> mat(srcRow, srcCol);
    const size_t matSize = sizeof(Elem_T) * mat.GetElemNum();
    file.read(reinterpret_cast<char *>(mat.ptr), matSize);
    file.close();

    return *this = mat;
}
template <typename Elem_T>
void Mat_T<Elem_T>::SaveToDisk(const char *path) const {
    const auto openMode = std::ios_base::binary | std::ios_base::out;
    std::ofstream file(path, openMode);

    file.write(reinterpret_cast<const char *>(&row), sizeof(row));
    file.write(reinterpret_cast<const char *>(&col), sizeof(col));

    const size_t matSize = sizeof(Elem_T) * GetElemNum();
    file.write(reinterpret_cast<const char *>(ptr), matSize);
    file.close();
}

template <typename Elem_T>
void Mat_T<Elem_T>::PrintLine(uint32_t row, int width, std::ostream &outStream) const {
    if (row >= this->row) return;
    constexpr uint32_t maxCol = 8U;
    const Elem_T *lineArr = ptr + row * this->col;

    if (this->col <= maxCol) {
        for (uint32_t i = 0U; i < this->col; ++i) {
            outStream << std::setw(width) << lineArr[i] << ' ';
        }
        outStream << std::endl;
    } else {
        for (uint32_t i = 0U; i < maxCol / 2U; ++i) {
            outStream << std::setw(width) << lineArr[i] << ' ';
        }
        outStream << "...";

        for (uint32_t i = this->col - maxCol / 2U; i < this->col; ++i) {
            outStream << std::setw(width) << lineArr[i] << ' ';
        }
        outStream << std::endl;
    }
}

template <typename Elem_T>
void Mat_T<Elem_T>::Print(int width, std::ostream &outStream) const {
    constexpr uint32_t maxRow = 10U;
    if (row <= maxRow) {
        for (uint32_t i = 0U; i < row; ++i) {
            PrintLine(i, width, outStream);
        }
    } else {
        for (uint32_t i = 0U; i < maxRow / 2U; ++i) {
            PrintLine(i, width, outStream);
        }
        outStream << "    ... ... ... ..." << std::endl;
        for (uint32_t i = row - maxRow / 2U; i < row; ++i) {
            PrintLine(i, width, outStream);
        }
    }
}

template <typename Elem_T>
std::ostream &operator<<(std::ostream &outStream, const Mat_T<Elem_T> &mat) {
    mat.Print(12, outStream);
    return outStream;
}

}  // namespace Matrix

// === function ====== function ====== function ====== function ===

// === function ====== function ====== function ====== function ===

// === function ====== function ====== function ====== function ===

namespace Matrix {

template <typename Ret_T, typename Elem_T1, typename Elem_T2>
Ret_T ValueAdd(Elem_T1 e1, Elem_T2 e2) noexcept {
    return static_cast<Ret_T>(e1 + e2);
}
template <typename Ret_T, typename Elem_T1, typename Elem_T2>
Ret_T ValueSub(Elem_T1 e1, Elem_T2 e2) noexcept {
    return static_cast<Ret_T>(e1 - e2);
}
template <typename Ret_T, typename Elem_T1, typename Elem_T2>
Ret_T ValueMul(Elem_T1 e1, Elem_T2 e2) noexcept {
    return static_cast<Ret_T>(e1 * e2);
}
template <typename Ret_T, typename Elem_T1, typename Elem_T2>
Ret_T ValueDiv(Elem_T1 e1, Elem_T2 e2) noexcept {
    return static_cast<Ret_T>(e1 / e2);
}

template <typename Ret_T, typename T1, typename T2, typename Func_T>
void OpEachElem(Mat_T<Ret_T> &ret, const Mat_T<T1> &mat1, const Mat_T<T2> &mat2,
                Func_T func) noexcept {
    assert(ret.row == mat1.row && ret.col == mat1.col);
    assert(mat1.row == mat2.row && mat1.col == mat2.col);

    const uint32_t elemNum = ret.GetElemNum();
    for (uint32_t i = 0U; i < elemNum; ++i) {
        ret.ptr[i] = func(mat1.ptr[i], mat2.ptr[i]);
    }
}
template <typename Ret_T, typename T1, typename T2, typename Func_T>
void OpEachElem(Mat_T<Ret_T> &ret, const Mat_T<T1> &mat1, T2 value, Func_T func) noexcept {
    assert(ret.row == mat1.row && ret.col == mat1.col);

    const uint32_t elemNum = ret.GetElemNum();
    for (uint32_t i = 0U; i < elemNum; ++i) {
        ret.ptr[i] = func(mat1.ptr[i], value);
    }
}
template <typename Ret_T, typename T1, typename T2, typename Func_T>
void OpEachElem(Mat_T<Ret_T> &ret, T2 value, const Mat_T<T1> &mat1, Func_T func) noexcept {
    assert(ret.row == mat1.row && ret.col == mat1.col);

    const uint32_t elemNum = ret.GetElemNum();
    for (uint32_t i = 0U; i < elemNum; ++i) {
        ret.ptr[i] = func(value, mat1.ptr[i]);
    }
}
template <typename Ret_T, typename T1, typename Func_T>
void OpEachElem(Mat_T<Ret_T> &ret, const Mat_T<T1> &mat1, Func_T func) noexcept {
    assert(ret.row == mat1.row && ret.col == mat1.col);

    const uint32_t elemNum = ret.GetElemNum();
    for (uint32_t i = 0U; i < elemNum; ++i) {
        ret.ptr[i] = func(mat1.ptr[i]);
    }
}
template <typename Ret_T, typename T1, typename T2>
void MatMul(Mat_T<Ret_T> &ret, const Mat_T<T1> &mat1, const Mat_T<T2> &mat2) noexcept {
    assert(mat1.col == mat2.row);
    std::fill_n(ret.ptr, ret.GetElemNum(), Ret_T{});

    for (uint32_t k = 0U; k < mat1.col; ++k) {
        for (uint32_t i = 0U; i < ret.row; ++i) {
            const T1 src1 = mat1.ptr[i * mat1.col + k];
            Ret_T *destPtr = &ret.ptr[i * ret.col];

            for (uint32_t j = 0U; j < ret.col; ++j) {
                destPtr[j] += src1 * mat2.ptr[k * mat2.col + j];
            }
        }
    }
}

template <typename Ret_T, typename T1, typename T2>
Ret_T VectorMul(const T1 *srcA, const T2 *srcB, uint32_t num) noexcept {
    Ret_T retValue{};
    for (uint32_t i = 0U; i < num; ++i) {
        retValue += srcA[i] * srcB[i];
    }
    return retValue;
}

template <typename Ret_T, typename Elem_T1, typename Elem_T2>
Ret_T DotMul(const Mat_T<Elem_T1> &mat1, const Mat_T<Elem_T2> &mat2) {
    const uint32_t elemNum = mat1.GetElemNum();
    assert(mat1.GetDim() == 1 && mat2.GetDim() == 1 && elemNum == mat2.GetElemNum());
    return VectorMul<Ret_T>(mat1.ptr, mat2.ptr, elemNum);
}

template <typename Ret_T, typename Elem_T1, typename Elem_T2>
Mat_T<Ret_T> MatDotMul(const Mat_T<Elem_T1> &mat1, const Mat_T<Elem_T2> &mat2) {
    assert(mat2.GetDim() == 1 && mat1.col == mat2.GetElemNum());
    Mat_T<Ret_T> retMat(mat1.row);

    for (uint32_t i = 0U; i < mat1.row; ++i) {
        retMat.ptr[i] = VectorMul<Ret_T>(&mat1.ptr[i * mat1.col], mat2.ptr, mat1.col);
    }
    return retMat;
}

template <typename Ret_T, typename Elem_T1, typename Elem_T2>
Mat_T<Ret_T> OutMul(const Mat_T<Elem_T1> &mat1, const Mat_T<Elem_T2> &mat2) {
    const uint32_t elemNum1 = mat1.GetElemNum();
    const uint32_t elemNum2 = mat2.GetElemNum();
    Mat_T<Ret_T> retMat(elemNum1, elemNum2);

    for (uint32_t i = 0U; i < elemNum1; ++i) {
        for (uint32_t j = 0U; j < elemNum2; ++j) {
            retMat.ptr[i * retMat.col + j] = mat1.ptr[i] * mat2.ptr[j];
        }
    }
    return retMat;
}

template <typename Elem_T>
Mat_T<Elem_T> Diagonal(const Mat_T<Elem_T> &mat) {
    assert(mat.GetDim() == 1);
    const uint32_t elemNum = mat.GetElemNum();
    Mat_T<Elem_T> ret(elemNum, elemNum);

    std::fill_n(ret.ptr, elemNum * elemNum, Elem_T{});

    for (uint32_t i = 0U; i < elemNum; ++i) {
        ret.ptr[i * elemNum + i] = mat.ptr[i];
    }
    return ret;
}

template <typename Elem_T>
Mat_T<Elem_T> tanh(const Mat_T<Elem_T> &mat) noexcept {
    Mat_T<Elem_T> ret(mat.row, mat.col);
    const uint32_t elemNum = mat.GetElemNum();

    for (uint32_t i = 0U; i < elemNum; ++i) {
        ret.ptr[i] = std::tanh(mat.ptr[i]);
    }
    return ret;
}

template <typename Elem_T>
Mat_T<Elem_T> softmax(const Mat_T<Elem_T> &mat) noexcept {
    Mat_T<Elem_T> ret(mat.row, mat.col);
    const uint32_t elemNum = mat.GetElemNum();
    Elem_T sumValue = Elem_T{};

    for (uint32_t i = 0U; i < elemNum; ++i) {
        ret.ptr[i] = std::exp(mat.ptr[i]);
        sumValue += ret.ptr[i];
    }
    for (uint32_t i = 0U; i < elemNum; ++i) {
        ret.ptr[i] /= sumValue;
    }
    return ret;
}

template <typename Elem_T>
Mat_T<Elem_T> d_tanh(const Mat_T<Elem_T> &mat) noexcept {
    const uint32_t elemNum = mat.GetElemNum();
    Mat_T<Elem_T> ret(elemNum, elemNum);
    std::fill_n(ret.ptr, elemNum * elemNum, Elem_T{});

    for (uint32_t i = 0U; i < elemNum; ++i) {
        const Elem_T temp = std::cosh(mat.ptr[i]);
        ret.ptr[i * elemNum + i] = 1 / temp / temp;
    }
    return ret;
}

template <typename Elem_T>
Mat_T<Elem_T> d_softmax(const Mat_T<Elem_T> &mat) noexcept {
    Mat_T<Elem_T> ret = softmax(mat);
    return Diagonal(ret) - OutMul<Elem_T>(ret, ret);
}

}  // namespace Matrix

// === operator method ====== operator method ====== operator method ====== operator method ===

// === operator method ====== operator method ====== operator method ====== operator method ===

// === operator method ====== operator method ====== operator method ====== operator method ===

namespace Matrix {

// === operate with another matrix ====== operate with another matrix ===

// === operate with another matrix ====== operate with another matrix ===

template <typename T1, typename T2>
Mat_T<T1> operator+(const Mat_T<T1> &mat1, const Mat_T<T2> &mat2) {
    Mat_T<T1> ret(mat1.row, mat1.col);
    OpEachElem<T1>(ret, mat1, mat2, ValueAdd<T1, T1, T2>);
    return ret;
}
template <typename T1, typename T2>
Mat_T<T1> operator-(const Mat_T<T1> &mat1, const Mat_T<T2> &mat2) {
    Mat_T<T1> ret(mat1.row, mat1.col);
    OpEachElem<T1>(ret, mat1, mat2, ValueSub<T1, T1, T2>);
    return ret;
}
template <typename T1, typename T2>
Mat_T<T1> operator*(const Mat_T<T1> &mat1, const Mat_T<T2> &mat2) {
    Mat_T<T1> ret(mat1.row, mat2.col);
    MatMul(ret, mat1, mat2);
    return ret;
}

// === operate with another value ====== operate with another value ===

// === operate with another value ====== operate with another value ===

template <typename T1, typename T2>
Mat_T<T2> operator+(const Mat_T<T1> &mat1, T2 value) {
    Mat_T<T2> ret(mat1.row, mat1.col);
    OpEachElem<T2>(ret, mat1, value, ValueAdd<T2, T1, T2>);
    return ret;
}
template <typename T1, typename T2>
Mat_T<T2> operator-(const Mat_T<T1> &mat1, T2 value) {
    Mat_T<T2> ret(mat1.row, mat1.col);
    OpEachElem<T2>(ret, mat1, value, ValueSub<T2, T1, T2>);
    return ret;
}
template <typename T1, typename T2>
Mat_T<T2> operator*(const Mat_T<T1> &mat1, T2 value) {
    Mat_T<T2> ret(mat1.row, mat1.col);
    OpEachElem<T2>(ret, mat1, value, ValueMul<T2, T1, T2>);
    return ret;
}
template <typename T1, typename T2>
Mat_T<T2> operator/(const Mat_T<T1> &mat1, T2 value) {
    Mat_T<T2> ret(mat1.row, mat1.col);
    OpEachElem<T2>(ret, mat1, value, ValueDiv<T2, T1, T2>);
    return ret;
}

template <typename T1, typename T2>
Mat_T<T2> operator+(T2 value, const Mat_T<T1> &mat1) {
    return mat1 + value;
}
template <typename T1, typename T2>
Mat_T<T2> operator-(T2 value, const Mat_T<T1> &mat1) {
    Mat_T<T2> ret(mat1.row, mat1.col);
    OpEachElem<T2>(ret, value, mat1, ValueSub<T2, T1, T2>);
    return ret;
}
template <typename T1, typename T2>
Mat_T<T2> operator*(T2 value, const Mat_T<T1> &mat1) {
    return mat1 * value;
}
template <typename T1, typename T2>
Mat_T<T2> operator/(T2 value, const Mat_T<T1> &mat1) {
    Mat_T<T2> ret(mat1.row, mat1.col);
    OpEachElem<T2>(ret, value, mat1, ValueDiv<T2, T1, T2>);
    return ret;
}

// === self operation with another mat ====== self operation with another mat ===

// === self operation with another mat ====== self operation with another mat ===

template <typename T1, typename T2>
Mat_T<T1> &operator+=(Mat_T<T1> &mat1, const Mat_T<T2> &mat2) noexcept {
    OpEachElem<T1>(mat1, mat1, mat2, ValueAdd<T1, T1, T2>);
    return mat1;
}
template <typename T1, typename T2>
Mat_T<T1> &operator-=(Mat_T<T1> &mat1, const Mat_T<T2> &mat2) noexcept {
    OpEachElem<T1>(mat1, mat1, mat2, ValueSub<T1, T1, T2>);
    return mat1;
}

// === self operation with another value ====== self operation with another value ===

// === self operation with another value ====== self operation with another value ===

template <typename T1, typename T2>
Mat_T<T1> &operator+=(Mat_T<T1> &mat1, T2 value) noexcept {
    OpEachElem<T1>(mat1, mat1, value, ValueAdd<T1, T1, T2>);
    return mat1;
}
template <typename T1, typename T2>
Mat_T<T1> &operator-=(Mat_T<T1> &mat1, T2 value) noexcept {
    OpEachElem<T1>(mat1, mat1, value, ValueSub<T1, T1, T2>);
    return mat1;
}
template <typename T1, typename T2>
Mat_T<T1> &operator*=(Mat_T<T1> &mat1, T2 value) noexcept {
    OpEachElem<T1>(mat1, mat1, value, ValueMul<T1, T1, T2>);
    return mat1;
}
template <typename T1, typename T2>
Mat_T<T1> &operator/=(Mat_T<T1> &mat1, T2 value) noexcept {
    OpEachElem<T1>(mat1, mat1, value, ValueDiv<T1, T1, T2>);
    return mat1;
}

}  // namespace Matrix

#endif
