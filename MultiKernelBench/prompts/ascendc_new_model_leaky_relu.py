project_json_src='''
[
    {
        "op": "LeakyReluCustom",
        "language": "cpp",
        "input_desc": [
            {
                "name": "x",
                "param_type": "required",
                "format": [
                    "ND"
                ],
                "type": [
                    "float"
                ]
            }
        ],
        "output_desc": [
            {
                "name": "y",
                "param_type": "required",
                "format": [
                    "ND"
                ],
                "type": [
                    "float"
                ]
            }
        ],
        "attr": [
            {
                "name": "negative_slope",
                "param_type": "optional",
                "type": "float",
                "default_value": "0.1"
            }
        ]
    }
]

'''

host_tiling_src="""
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(LeakyReluCustomTilingData)
TILING_DATA_FIELD_DEF(uint32_t, totalLength);
TILING_DATA_FIELD_DEF(uint32_t, tileNum);
TILING_DATA_FIELD_DEF(float, negativeSlope);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(LeakyReluCustom, LeakyReluCustomTilingData)
}
"""

host_operator_src="""

#include "leaky_relu_custom_tiling.h"
#include "register/op_def_registry.h"


namespace optiling {
const uint32_t BLOCK_DIM = 8;
const uint32_t TILE_NUM = 16;

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    LeakyReluCustomTilingData tiling;
    uint32_t totalLength = context->GetInputShape(0)->GetOriginShape().GetShapeSize();
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    const float *negativeSlope = attrs->GetAttrPointer<float>(0);

    context->SetBlockDim(BLOCK_DIM);
    tiling.set_totalLength(totalLength);
    tiling.set_tileNum(TILE_NUM);
    tiling.set_negativeSlope(*negativeSlope);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext *context)
{
    const gert::Shape *x1_shape = context->GetInputShape(0);
    gert::Shape *y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}
static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    const ge::DataType x1_dtype = context->GetInputDataType(0);
    context->SetOutputDataType(0, x1_dtype);
    return GRAPH_SUCCESS;
}
} // namespace ge


namespace ops {
class LeakyReluCustom : public OpDef {
public:
    explicit LeakyReluCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("negative_slope").AttrType(OPTIONAL).Float(0.1);

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");

    }
};

OP_ADD(LeakyReluCustom);
}

"""

kernel_src="""
#include "kernel_operator.h"
constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue

class KernelLeakyRelu {
public:
    __aicore__ inline KernelLeakyRelu() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength, uint32_t tileNum, float negativeSlope)
    {
        this->blockLength = totalLength / AscendC::GetBlockNum();
        this->tileNum = tileNum;
        this->negativeSlope = static_cast<float>(negativeSlope);
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM;

        // get start index for current core, core parallel
        xGm.SetGlobalBuffer((__gm__ float *)x + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        yGm.SetGlobalBuffer((__gm__ float *)y + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        // pipe alloc memory to queue, the unit is Bytes
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(tmpBuffer1, this->tileLength * sizeof(float));
        pipe.InitBuffer(tmpBuffer2, this->tileLength * sizeof(float));
    }
    __aicore__ inline void Process()
    {
        // loop count need to be doubled, due to double buffer
        int32_t loopCount = this->tileNum * BUFFER_NUM;
        // tiling strategy, pipeline parallel
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        // alloc tensor from queue memory
        AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        // copy progress_th tile from global tensor to local tensor
        AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
        // enque input tensors to VECIN queue
        inQueueX.EnQue(xLocal);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        // deque input tensors from VECIN queue
        AscendC::LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        AscendC::LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
        AscendC::LocalTensor<float> tmpTensor1 = tmpBuffer1.Get<float>();
        AscendC::LocalTensor<float> tmpTensor2 = tmpBuffer2.Get<float>();
        float inputVal = 0.0;
        AscendC::Maxs(tmpTensor1, xLocal, inputVal, this->tileLength);
        AscendC::Mins(tmpTensor2, xLocal, inputVal, this->tileLength);
        AscendC::Muls(tmpTensor2, tmpTensor2, this->negativeSlope, this->tileLength);
        AscendC::Add(yLocal, tmpTensor1, tmpTensor2, this->tileLength);
        // enque the output tensor to VECOUT queue
        outQueueY.EnQue<float>(yLocal);
        // free input tensors for reuse
        inQueueX.FreeTensor(xLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        // deque output tensor from VECOUT queue
        AscendC::LocalTensor<float> yLocal = outQueueY.DeQue<float>();
        // copy progress_th tile from local tensor to global tensor
        AscendC::DataCopy(yGm[progress * this->tileLength], yLocal, this->tileLength);
        // free output tensor for reuse
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpBuffer1, tmpBuffer2;
    // create queues for input, in this case depth is equal to buffer num
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX;
    // create queue for output, in this case depth is equal to buffer num
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::GlobalTensor<float> xGm, yGm;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
    float negativeSlope;
};

extern "C" __global__ __aicore__ void leaky_relu_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelLeakyRelu op;
    op.Init(x, y, tiling_data.totalLength, tiling_data.tileNum, tiling_data.negativeSlope);
    op.Process();
}
"""

python_bind_src="""
#include <torch/library.h>
#include <torch/csrc/autograd/custom_function.h>
#include "pytorch_npu_helper.hpp"
#include <torch/extension.h>

at::Tensor leaky_relu_impl_npu(const at::Tensor& self, double negative_slope) {
    // float argument not supported now, so use double negative_slope
    at::Tensor result = at::empty_like(self);
    EXEC_NPU_CMD(aclnnLeakyReluCustom, self, negative_slope, result);
    return result;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("leaky_relu_custom", &leaky_relu_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("leaky_relu_custom", &leaky_relu_impl_npu, "LeakyReLU activation");
}
"""

model_src='''
import torch
import torch_npu
import custom_ops_lib
class ModelNew(torch.nn.Module):
    def __init__(self, negative_slope: float = 0.1):
        super(ModelNew, self).__init__()
        self.negative_slope = negative_slope
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return custom_ops_lib.leaky_relu_custom(x, self.negative_slope)
'''
