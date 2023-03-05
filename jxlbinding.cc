#include <stdio.h>
#include <errno.h>

#include <iostream>
#include <thread>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <Python.h>

#include "libjxl/lib/extras/dec/decode.h"
#include "libjxl/lib/extras/enc/jxl.h"
#include "libjxl/lib/extras/packed_image.h"

#include "libjxl/tools/file_io.h"

using namespace std;

namespace py = pybind11;


void get_compressed(JxlEncoder* encoder, vector<uint8_t>* compressed) {
    compressed->resize(64);
    uint8_t* next_out = compressed->data();
    size_t avail_out = compressed->size() - (next_out - compressed->data());
    JxlEncoderStatus process_result = JXL_ENC_NEED_MORE_OUTPUT;
    while (process_result == JXL_ENC_NEED_MORE_OUTPUT) {
        process_result = JxlEncoderProcessOutput(encoder, &next_out, &avail_out);
        if (process_result == JXL_ENC_NEED_MORE_OUTPUT) {
        size_t offset = next_out - compressed->data();
        compressed->resize(compressed->size() * 2);
        next_out = compressed->data() + offset;
        avail_out = compressed->size() - offset;
        }
    }
    compressed->resize(next_out - compressed->data());
}


void encode_image(py::array_t<uint16_t, py::array::c_style> image, string filepath) {

    JxlPixelFormat format = {3, JXL_TYPE_UINT16, JXL_NATIVE_ENDIAN, 0};
    JxlEncoder* encoder = JxlEncoderCreate(nullptr);
    JxlBasicInfo basic_info;
    JxlEncoderInitBasicInfo(&basic_info);
    basic_info.xsize = image.shape()[1];
    basic_info.ysize = image.shape()[0];
    basic_info.bits_per_sample = 16;
    basic_info.exponent_bits_per_sample = 0;
    basic_info.uses_original_profile = JXL_FALSE;
    JxlEncoderSetBasicInfo(encoder, &basic_info);
    JxlEncoderFrameSettings* settings = JxlEncoderFrameSettingsCreate(encoder, nullptr);
    JxlEncoderFrameSettingsSetOption(settings, JXL_ENC_FRAME_SETTING_EFFORT, 1);

    const void *buffer = reinterpret_cast<const void*>(image.data());
    if (JXL_ENC_SUCCESS != JxlEncoderAddImageFrame(
        settings,
        &format,
        buffer, 
        size_t(image.shape()[0] * image.shape()[1] * image.shape()[2] * 2))) {

        cout << "encoder failed..." << endl;
    }
    JxlEncoderCloseFrames(encoder);

    vector<uint8_t> compressed;
    get_compressed(encoder, &compressed);

    if (!jpegxl::tools::WriteFile(filepath.c_str(), compressed)) {
        cout << "write failed..." << endl;
    }
}


void encode_images_sub(vector<py::array_t<uint16_t, py::array::c_style>> images, vector<string> filepaths) {
    int batch_size = images.size();

    for (int i=0; i<batch_size; i++) {
        encode_image(images[i], filepaths[i]);
    }
}


void encode_images_super(vector<vector<py::array_t<uint16_t, py::array::c_style>>> images, vector<vector<string>> filepaths) {

    int batch_size = images.size();

    vector<thread> threads(batch_size);

    for (int i=0; i<batch_size; i++) {
        threads[i] = thread(encode_images_sub, images[i], filepaths[i]);
    }

    for (int i=0; i<batch_size; i++) {
        threads[i].join();
    }
}

PYBIND11_MODULE(jxlbinding, m) {
    m.doc() = "jxl binding"; // optional module docstring

    m.def("encode_image", &encode_image, "save jxl image");
    m.def("encode_images_sub", &encode_images_sub, "save list of jxl images");
    m.def("encode_images_super", &encode_images_super, "save list of lists of jxl images concurrently");
}