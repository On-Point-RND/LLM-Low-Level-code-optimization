project_json_src='''
[
    {
        "op":"LayerNormCustom",
        "input_desc":[
            {
                "name":"x",
                "format": [
                    "ND"
                ],
                "type": [
                    "float"
                ],
                "param_type":"required"
            },
            {
                "name":"gamma",
                "format": [
                    "ND"
                ],
                "type": [
                    "float"
                ],
                "param_type":"required"
            },
            {
                "name":"beta",
                "format": [
                    "ND"
                ],
                "type": [
                    "float"
                ],
                "param_type":"required"
            }
        ],
        "output_desc":[
            {
                "name":"res_out",
                "format": [
                    "ND"
                ],
                "type": [
                    "float"
                ],
                "param_type":"required"
            }
        ],
        "attr":[
            {
                "name":"epsilon",
                "type": [
                    "float"
                ],
                "default_value":0.00001,
                "param_type":"optional"
            }
        ]
    }
]
'''

host_tiling_src="""
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(LayerNormCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, rowNum);
  TILING_DATA_FIELD_DEF(uint32_t, rowNumSp);
  TILING_DATA_FIELD_DEF(uint32_t, rowLength);
  TILING_DATA_FIELD_DEF(uint32_t, blockPivot);
  TILING_DATA_FIELD_DEF(uint32_t, tileLoop);
  TILING_DATA_FIELD_DEF(uint32_t, tileLength);
  TILING_DATA_FIELD_DEF(uint32_t, loopCount);
  TILING_DATA_FIELD_DEF(float, factor);
  TILING_DATA_FIELD_DEF(float, mfactor);
  TILING_DATA_FIELD_DEF(float, eps);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(LayerNormCustom, LayerNormCustomTilingData)
}
"""

host_operator_src="""
#include "layer_norm_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
const uint32_t BLOCK_DIM = 48;

static ge::graphStatus TilingFunc(gert::TilingContext* context) {
  LayerNormCustomTilingData tiling;
  const gert::StorageShape* x1_shape = context->GetInputShape(0);
  const gert::Shape shape = x1_shape->GetStorageShape();
  auto rowNum = shape.GetDim(0);
  auto colNum = shape.GetDim(1);
  auto coreRowNum = rowNum / BLOCK_DIM;

  const float* epsilonAttr = context->GetAttrs()->GetAttrPointer<float>(0);
  tiling.set_eps(*epsilonAttr);
  tiling.set_rowNum(coreRowNum);
  tiling.set_rowNumSp(coreRowNum + 1);
  tiling.set_rowLength(colNum);
  tiling.set_blockPivot(rowNum - coreRowNum * BLOCK_DIM);
  uint32_t tileLoop = 18;
  tiling.set_tileLoop(tileLoop);
  tiling.set_tileLength(tileLoop * colNum);
  tiling.set_loopCount(coreRowNum / tileLoop);
  tiling.set_factor(1.0f / colNum);
  tiling.set_mfactor(-1.0f / colNum);

  context->SetBlockDim(BLOCK_DIM);
  tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                      context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
  context->SetTilingKey(1);

  return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context) {
  const gert::Shape* x1_shape = context->GetInputShape(0);
  gert::Shape* y_shape = context->GetOutputShape(0);
  *y_shape = *x1_shape;
  return GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class LayerNormCustom : public OpDef {
 public:
  explicit LayerNormCustom(const char* name) : OpDef(name) {
    this->Input("x")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND});
    this->Input("gamma")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND});
    this->Input("beta")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND});
    this->Output("res_out")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND});
    this->Attr("epsilon").AttrType(OPTIONAL).Float(1e-05);

    this->SetInferShape(ge::InferShape);

    this->AICore().SetTiling(optiling::TilingFunc);
    this->AICore().AddConfig("ascend910b");
  }
};

OP_ADD(LayerNormCustom);
}
"""

kernel_src="""
#include "kernel_operator.h"
constexpr int32_t BUFFER_NUM = 1;

class KernelLayerNorm {
 public:
  __aicore__ inline KernelLayerNorm() {}
  __aicore__ inline void InitTiling(GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    rowNum = tiling_data.rowNum;
    rowNumSp = tiling_data.rowNumSp;
    rowLength = tiling_data.rowLength;
    blockPivot = tiling_data.blockPivot;
    tileLoop = tiling_data.tileLoop;
    tileLength = tiling_data.tileLength;
    loopCount = tiling_data.loopCount;
    factor = tiling_data.factor;
    mfactor = tiling_data.mfactor;
    eps = tiling_data.eps;
  }
  __aicore__ inline void Init(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR z,
                              GM_ADDR tiling) {
    InitTiling(tiling);

    this->leftRow = this->rowNum % this->tileLoop;
    if (AscendC::GetBlockIdx() < this->blockPivot) {
      this->rowNum = this->rowNumSp;
      this->leftRow += 1;
    }

    this->blockLength = this->rowNum * this->rowLength;
    uint32_t offset = 0;
    if (AscendC::GetBlockIdx() < this->blockPivot) {
      offset = this->blockLength * AscendC::GetBlockIdx();
    } else {
      offset = this->blockLength * AscendC::GetBlockIdx() +
               this->rowLength * this->blockPivot;
    }

    xGm.SetGlobalBuffer((__gm__ float *)x + offset, this->blockLength);
    zGm.SetGlobalBuffer((__gm__ float *)z + offset, this->blockLength);

    gammaGm.SetGlobalBuffer((__gm__ float *)gamma, this->rowLength);
    betaGm.SetGlobalBuffer((__gm__ float *)beta, this->rowLength);

    pipe.InitBuffer(queueX, BUFFER_NUM, this->tileLength * sizeof(float));
    pipe.InitBuffer(queueZ, BUFFER_NUM, this->tileLength * sizeof(float));

    pipe.InitBuffer(tmpBuffer1, 64 * sizeof(float));
    pipe.InitBuffer(tmpBuffer2, 64 * sizeof(float));
    pipe.InitBuffer(onesBuffer, 64 * sizeof(float));

    pipe.InitBuffer(queueGamma, 1, this->rowLength * sizeof(float));
    pipe.InitBuffer(queueBeta, 1, this->rowLength * sizeof(float));
  }
  __aicore__ inline void Process() {
    for (int32_t i = 0; i < this->loopCount; i++) {
      CopyIn(i, this->tileLoop);
      Compute(i, this->tileLoop);
      CopyOut(i, this->tileLoop);
    }
    if (this->leftRow > 0) {
      CopyIn(this->loopCount, this->leftRow);
      Compute(this->loopCount, this->leftRow);
      CopyOut(this->loopCount, this->leftRow);
    }
  }

 private:
  __aicore__ inline void CopyIn(int32_t progress, int32_t rowNum) {
    AscendC::LocalTensor<float> xLocal = queueX.AllocTensor<float>();
    AscendC::LocalTensor<float> gammaLocal = queueGamma.AllocTensor<float>();
    AscendC::LocalTensor<float> betaLocal = queueBeta.AllocTensor<float>();
    AscendC::DataCopy(xLocal, xGm[progress * this->tileLength],
             this->rowLength * rowNum);
    AscendC::DataCopy(gammaLocal, gammaGm[0], this->rowLength);
    AscendC::DataCopy(betaLocal, betaGm[0], this->rowLength);
    queueX.EnQue(xLocal);
    queueGamma.EnQue(gammaLocal);
    queueBeta.EnQue(betaLocal);
  }

  __aicore__ inline void Compute(int32_t progress, int32_t rowNum) {
    AscendC::LocalTensor<float> xLocal = queueX.DeQue<float>();
    AscendC::LocalTensor<float> gammaLocal = queueGamma.DeQue<float>();
    AscendC::LocalTensor<float> betaLocal = queueBeta.DeQue<float>();

    AscendC::LocalTensor<float> tmpTensor1 = tmpBuffer1.Get<float>();
    AscendC::LocalTensor<float> tmpTensor2 = tmpBuffer2.Get<float>();
    AscendC::LocalTensor<float> onesLocal = onesBuffer.Get<float>();
    AscendC::LocalTensor<float> zLocal = queueZ.AllocTensor<float>();
    AscendC::Duplicate<float>(onesLocal, 1.0f, this->tileLoop);

    for (uint32_t j = 0; j < rowNum; ++j) {
      uint32_t buffIndex = j * this->rowLength;
      AscendC::ReduceSum<float>(tmpTensor2[j], xLocal[buffIndex], tmpTensor1,
                       this->rowLength);
    }

    AscendC::Muls(zLocal, tmpTensor2, this->mfactor, rowNum);

    for (uint32_t j = 0; j < rowNum; ++j) {
      uint32_t buffIndex = j * this->rowLength;
      AscendC::Adds(xLocal[buffIndex], xLocal[buffIndex], zLocal.GetValue(j),
           this->rowLength);
    }

    for (uint32_t j = 0; j < rowNum; ++j) {
      uint32_t buffIndex = j * this->rowLength;
      AscendC::Mul(zLocal[buffIndex], xLocal[buffIndex], xLocal[buffIndex],
          this->rowLength);
    }
    for (uint32_t j = 0; j < rowNum; ++j) {
      uint32_t buffIndex = j * this->rowLength;
      AscendC::ReduceSum<float>(tmpTensor2[j], zLocal[buffIndex], tmpTensor1,
                       this->rowLength);
    }
    AscendC::Muls(tmpTensor2, tmpTensor2, this->factor, rowNum);
    AscendC::Adds(tmpTensor2, tmpTensor2, this->eps, rowNum);
    AscendC::Sqrt(tmpTensor2, tmpTensor2, rowNum);
    AscendC::Div(tmpTensor2, onesLocal, tmpTensor2, rowNum);

    for (uint32_t j = 0; j < rowNum; ++j) {
      uint32_t buffIndex = j * this->rowLength;
      AscendC::Muls(zLocal[buffIndex], xLocal[buffIndex], tmpTensor2.GetValue(j),
           this->rowLength);
    }

    for (uint32_t j = 0; j < rowNum; ++j) {
      uint32_t buffIndex = j * this->rowLength;
      AscendC::Mul(zLocal[buffIndex], zLocal[buffIndex], gammaLocal, this->rowLength);
      AscendC::Add(zLocal[buffIndex], zLocal[buffIndex], betaLocal, this->rowLength);
    }

    queueZ.EnQue<float>(zLocal);
    queueGamma.FreeTensor(gammaLocal);
    queueBeta.FreeTensor(betaLocal);
    queueX.FreeTensor(xLocal);
  }

  __aicore__ inline void CopyOut(int32_t progress, int32_t rowNum) {
    AscendC::LocalTensor<float> zLocal = queueZ.DeQue<float>();

    AscendC::DataCopy(zGm[progress * this->tileLength], zLocal,
             rowNum * this->rowLength);

    queueZ.FreeTensor(zLocal);
  }

 private:
  AscendC::TPipe pipe;
  AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpBuffer1, tmpBuffer2, onesBuffer;
  AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> queueX;
  AscendC::TQue<AscendC::QuePosition::VECIN, 1> queueGamma, queueBeta;
  AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> queueZ;
  AscendC::GlobalTensor<float> xGm;
  AscendC::GlobalTensor<float> gammaGm;
  AscendC::GlobalTensor<float> betaGm;
  AscendC::GlobalTensor<float> zGm;

  uint32_t blockLength = 0;
  uint32_t leftRow = 0;
  uint32_t rowNum = 341;
  uint32_t rowNumSp = 342;
  uint32_t rowLength = 1024;
  uint32_t blockPivot = 16;
  uint32_t tileLoop = 8;
  uint32_t tileLength = 8 * 1024;
  uint32_t loopCount = 42;
  float factor = 0.0009765625;
  float mfactor = -0.0009765625;
  float eps = 1e-5;
};
extern "C" __global__ __aicore__ void layer_norm_custom(
    GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR res_out, GM_ADDR workspace,
    GM_ADDR tiling) {
  KernelLayerNorm op;
  op.Init(x, gamma, beta, res_out, tiling);
  if (TILING_KEY_IS(1)) {
    op.Process();
  }
}
"""

python_bind_src="""
#include <torch/library.h>
#include <torch/csrc/autograd/custom_function.h>
#include "pytorch_npu_helper.hpp"
#include <torch/extension.h>

at::Tensor layer_norm_impl_npu(const at::Tensor& x, const at::Tensor& gamma, const at::Tensor& beta, double epsilon) {
    // float argument not supported now, so use double negative_slope
    at::Tensor result = at::empty_like(x);
    EXEC_NPU_CMD(aclnnLayerNormCustom, x, gamma, beta, epsilon, result);
    return result;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("layer_norm_custom", &layer_norm_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("layer_norm_custom", &layer_norm_impl_npu, "layer norm");
}
"""

model_src='''
import torch, torch_npu, custom_ops_lib
class ModelNew(torch.nn.Module):
    def __init__(self, normalized_shape: tuple):
        super(ModelNew, self).__init__()
        self.normalized_shape = normalized_shape
        self.epsilon = 1e-5
        self.gamma = torch.nn.Parameter(torch.ones(normalized_shape))
        self.beta = torch.nn.Parameter(torch.zeros(normalized_shape))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return custom_ops_lib.layer_norm_custom(x, self.gamma, self.beta, self.epsilon)
'''    