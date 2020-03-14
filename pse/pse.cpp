//
//  pse
//  reference https://github.com/whai362/PSENet/issues/15
//  Created by liuheng on 11/3/19.
//  Copyright © 2019年 liuheng. All rights reserved.
//
#include <queue>
#include "include/pybind11/pybind11.h"
#include "include/pybind11/numpy.h"
#include "include/pybind11/stl.h"
#include "include/pybind11/stl_bind.h"

namespace py = pybind11;

namespace pse{
    //S5->S0, small->big
    std::vector<std::vector<int32_t>> pse(
    py::array_t<int32_t, py::array::c_style> center,
    py::array_t<uint8_t, py::array::c_style> region)
    {
        auto pbuf_center = center.request();
        auto pbuf_region = region.request();
        if (pbuf_center.ndim != 2 || pbuf_center.shape[0]==0 || pbuf_center.shape[1]==0)
            throw std::runtime_error("label map must have a shape of (h>0, w>0)");
        int h = pbuf_center.shape[0];
        int w = pbuf_center.shape[1];
        if (pbuf_center.ndim != 2 || pbuf_region.shape[0]!=h || pbuf_region.shape[1]!=w)
            throw std::runtime_error("region must have a shape of (h>0, w>0)");

        std::vector<std::vector<int32_t>> res;
        for (size_t i = 0; i<h; i++)
            res.push_back(std::vector<int32_t>(w, 0));
        auto ptr_center = static_cast<int32_t *>(pbuf_center.ptr);
        auto ptr_region = static_cast<uint8_t *>(pbuf_region.ptr);

        std::queue<std::tuple<int, int, int32_t>> q, next_q;

        for (size_t i = 0; i<h; i++)
        {
            auto p_center = ptr_center + i*w;
            for(size_t j = 0; j<w; j++)
            {
                int32_t label = p_center[j];
                if (label>0)
                {
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
            int32_t l = std::get<2>(q_n);
            //store the edge pixel after one expansion
            for (int idx=0; idx<4; idx++)
            {
                int index_y = y + dy[idx];
                int index_x = x + dx[idx];
                if (index_y<0 || index_y>=h || index_x<0 || index_x>=w)
                    continue;
                if (!ptr_region[index_y*w+index_x] || res[index_y][index_x]>0)
                    continue;
                q.push(std::make_tuple(index_y, index_x, l));
                res[index_y][index_x]=l;
            }
        }

        return res;
    }
}

PYBIND11_MODULE(pse, m){
    m.def("pse_cpp", &pse::pse, " re-implementation pse algorithm(cpp)", py::arg("center"), py::arg("region"));
}