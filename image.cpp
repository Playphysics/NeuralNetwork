
#include <iostream>
#include <fstream>
#include "image.h"

namespace {

uint32_t GetU32(const char *data) {
    const uint8_t *ptr = reinterpret_cast<const uint8_t *>(data);
    uint32_t ret = 0U;
    for (int i = 0; i < 4; ++i) {
        ret |= static_cast<uint32_t>(ptr[i]) << ((3 - i) * 8);
    }
    return ret;
}

}  // namespace

namespace Image {

Img_T::Img_T(uint8_t labelValue, uint32_t row, uint32_t col, const uint8_t *data)
    : label(labelValue), imgRow(row), imgCol(col), mat(row * col, 1U) {
    std::copy_n(data, mat.GetElemNum(), mat.ptr);
}

void Img_T::Print() const {
    std::cout << "The label is " << static_cast<int>(label) << std::endl;

    for (uint32_t i = 0U; i < imgRow; ++i) {
        const ImgElem_T *imgPtr = &mat.ptr[i * imgCol];
        for (uint32_t j = 0U; j < imgCol; ++j) {
            switch (static_cast<int>(imgPtr[j] * 3)) {
                case 0: std::cout << "  "; break;
                case 1: std::cout << "HH"; break;
                default: std::cout << "##"; break;
            }
        }
        std::cout << std::endl;
    }
}

std::vector<Img_T> GetImgSet(const char *imgPath, const char *labelPath, uint32_t maxNum) {
    std::vector<Img_T> images;
    const auto openMode = std::ios_base::binary | std::ios_base::in;
    std::ifstream imgFile(imgPath, openMode);
    std::ifstream labFile(labelPath, openMode);

    char imgHeader[16] = {0};
    char labHeader[8] = {0};

    imgFile.read(imgHeader, sizeof(imgHeader));
    labFile.read(labHeader, sizeof(labHeader));

    uint32_t imgNum = GetU32(imgHeader + 4);
    const uint32_t imgRow = GetU32(imgHeader + 8);
    const uint32_t imgCol = GetU32(imgHeader + 12);

    if (imgNum == GetU32(labHeader + 4)) {
        if (imgNum > maxNum) imgNum = maxNum;
        const size_t imgSize = imgRow * imgCol;

        uint8_t *labData = new uint8_t[imgNum];
        uint8_t *buffer = new uint8_t[imgSize];

        images.reserve(imgNum);
        labFile.read(reinterpret_cast<char *>(labData), imgNum);

        for (uint32_t i = 0U; i < imgNum; ++i) {
            imgFile.read(reinterpret_cast<char *>(buffer), imgSize);
            images.push_back({labData[i], imgRow, imgCol, buffer});
            images[i].mat /= ColorMaxValue;
        }
        delete[] buffer;
        delete[] labData;
    } else {
        std::cerr << "The imageFile and labelFile don not match" << std::endl;
    }
    imgFile.close(), labFile.close();
    return images;
}

}  // namespace Image
