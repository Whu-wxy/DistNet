#include "include/pybind11/pybind11.h"
#include "include/pybind11/numpy.h"
#include "include/pybind11/stl.h"
#include "include/pybind11/stl_bind.h"
#include <iostream>
#include <queue>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;
namespace py = pybind11;

namespace dist{
    std::vector<std::vector<uint8_t>> dist(
    py::array_t<uint8_t, py::array::c_style> center,
    py::array_t<uint8_t, py::array::c_style> region,
    py::array_t<float, py::array::c_style> region_prob,
    float center_prob_threld, float region_prob_threld)
    {
        auto pbuf_center = center.request();
        auto pbuf_region = region.request();
        auto pbuf_region_prob = region_prob.request();
        if (pbuf_center.ndim != 2 || pbuf_center.shape[0]==0 || pbuf_center.shape[1]==0)
            throw std::runtime_error("center must have a shape of (h>0, w>0)");
        int h = pbuf_center.shape[0];
        int w = pbuf_center.shape[1];
        if (pbuf_region.ndim != 2 || pbuf_region.shape[0]!=h || pbuf_region.shape[1]!=w)
            throw std::runtime_error("region must have a shape of (h>0, w>0)");
        if (pbuf_region_prob.ndim != 2 || pbuf_region_prob.shape[0]!=h || pbuf_region_prob.shape[1]!=w)
            throw std::runtime_error("bi_region must have a shape of (h>0, w>0)");



        std::vector<std::vector<uint8_t>> res;
        for (size_t i = 0; i<h; i++)
            res.push_back(std::vector<uint8_t>(w, 0));
        auto ptr_center = static_cast<uint8_t *>(pbuf_center.ptr);  //0 or 1
        auto ptr_region = static_cast<uint8_t *>(pbuf_region.ptr);
        auto ptr_region_prob = static_cast<float *>(pbuf_region_prob.ptr);

        Mat matCenter = Mat::zeros(h, w, CV_8UC1);
        //convert to mat center
        for (int x = 0; x < h; ++x) {
            for (int y = 0; y < w; ++y) {
                matCenter.at<char>(x, y) = ptr_center[x * w + y];
            }
        }

        //convert to connectedComponents mat
        int label_num = connectedComponents(matCenter, matCenter, 4);

        int area[label_num + 1];
        memset(area, 0, sizeof(area));
        float area_prob[label_num + 1];
        memset(area_prob, 0, sizeof(area_prob));
        for (int x = 0; x < matCenter.rows; ++x) {
            for (int y = 0; y < matCenter.cols; ++y) {
                int label = matCenter.at<char>(x, y);
                if (label == 0) continue;
                area[label] += 1;
                area_prob[label] += ptr_region_prob[x * w + y];
            }
        }

        // area average prob
        for (int x = 0; x < label_num + 1; ++x){
            area_prob[x] = area_prob[x] / area[x];
            // filter area that prob less than center_prob_threld
            if (area_prob[x] < center_prob_threld){
                area[x] = 0;
            }
        }


        std::queue<std::tuple<int, int, uint8_t>> q, next_q;

        for (size_t i = 0; i<h; i++)
        {
            for(size_t j = 0; j<w; j++)
            {
                int label = matCenter.at<char>(i, j);
                if (label>0)
                {
                    if (area[label] < 5) {
                        continue;
                    }
                    q.push(std::make_tuple(i, j, label));
                    res[i][j] = label;
                }
            }
        }

        int dx[4] = {-1, 1, 0, 0};
        int dy[4] = {0, 0, -1, 1};
        //从上到下扫描式扩张
        // merge from small to large kernel progressively
        while(!q.empty())
        {
            //get each queue menber in q
            auto q_n = q.front();
            q.pop();
            int y = std::get<0>(q_n);
            int x = std::get<1>(q_n);
            uint8_t l = std::get<2>(q_n);
            //store the edge pixel after one expansion
            for (int idx=0; idx<4; idx++)
            {
                int index_y = y + dy[idx];
                int index_x = x + dx[idx];
                if (index_y<0 || index_y>=h || index_x<0 || index_x>=w)
                    continue;
                // ptr_region == 1 or 0
                if (!ptr_region[index_y*w+index_x] || res[index_y][index_x]>0)
                    continue;
                q.push(std::make_tuple(index_y, index_x, l));
                res[index_y][index_x]=l;
            }
        }
        // filter region area that prob less than threld
        float region_area_prob[label_num + 1];
        memset(region_area_prob, 0, sizeof(region_area_prob));
        int region_area[label_num + 1];
        memset(region_area, 0, sizeof(region_area));
        for (int x = 0; x < h; ++x) {
            for (int y = 0; y < w; ++y) {
                int label = res[x][y];
                if (label == 0) continue;
                region_area[label] += 1;
                region_area_prob[label] += ptr_region_prob[x * w + y];
            }
        }

        // area average prob
        for (int x = 0; x < label_num + 1; ++x){
            region_area_prob[x] = region_area_prob[x] / region_area[x];
            // filter area that prob less than region_prob_threld
            if (region_area_prob[x] < region_prob_threld){
                region_area[x] = 0;
            }
        }

        // [10,0,3,0,5,0]--->[10,1,1,3,2]
        int mark = 1;
        for (int x = 1; x < label_num + 1; ++x){
            int count = region_area[x];
            if (count == 0){
                continue;
            }
            region_area[x] = mark;
            mark += 1;
        }

        for (int x = 0; x < h; ++x) {
            for (int y = 0; y < w; ++y) {
                int label = res[x][y];
                if (label == 0) continue;
                res[x][y] = region_area[label];
            }
        }

        return res;
    }
}

PYBIND11_MODULE(dist, m){
    m.def("dist_cpp", &dist::dist, " rimplementation dist algorithm(cpp)", py::arg("center"), py::arg("region"), py::arg("region_prob"), py::arg("center_prob_threld"), py::arg("region_prob_threld"));
}
