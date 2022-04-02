#ifndef IMAGE_H_
#define IMAGE_H_

#include <cstdint>
#include <vector>
#include "matrix.hpp"

namespace Image {

typedef float ImgElem_T;
constexpr ImgElem_T ColorMaxValue = 255.0F;

struct Img_T final {
    uint8_t label;
    uint32_t imgRow, imgCol;
    Matrix::Mat_T<ImgElem_T> mat;

    Img_T(uint8_t labelValue, uint32_t row, uint32_t col, const uint8_t *data);
    void Print() const;
};

std::vector<Img_T> GetImgSet(const char *imgPath, const char *labelPath,
                             uint32_t maxNum = UINT32_MAX);

}  // namespace Image

#endif
