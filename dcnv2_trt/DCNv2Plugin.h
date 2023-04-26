/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __TRT_DCNV2_PLUGIN_H__
#define __TRT_DCNV2_PLUGIN_H__

#include <cstring>
#include <sstream>
#include <cassert>
#include <iostream>
#include <cuda.h>
#include <cudnn.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <string>
#include <vector>
#include <memory>
#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "NvInferRuntime.h"
#include "dcn_v2_im2col_cuda.h"
#include "macros.h"


namespace nvinfer1
{

	struct DCNv2Parameters
	{
		std::vector<int> dilation;
		std::vector<int> padding;
		std::vector<int> stride;
		int deformable_groups;
		int oc;
		int in_channel;
		int in_w;
		int in_h;
		int out_channel;
		int kernel_h;
		int kernel_w;
		int out_h;
		int out_w;
		float* mOne;
		float* mColumn;
	};

	const int DILATION_DIM = 2;
	const int PADDING_DIM = 2;
	const int STRIDE_DIM = 2;
	const int NUM_DCN_CHANNELS = 2;

	constexpr int N_DIM = 0;
	constexpr int C_DIM = 1;
	constexpr int H_DIM = 2;
	constexpr int W_DIM = 3;

	// inline unsigned int getElementSize(DataType t);

	class API DCNv2Plugin : public IPluginV2IOExt
	{
	public:
		DCNv2Plugin();

		DCNv2Plugin(const void* data, size_t length, const std::string& name);
		
		DCNv2Plugin(DCNv2Parameters param, const std::string& name);
		
		~DCNv2Plugin();

		IPluginV2IOExt* clone() const TRT_NOEXCEPT override;
		
		Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) TRT_NOEXCEPT override;
		
		bool supportsFormatCombination(int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) const TRT_NOEXCEPT override;
		
		void configurePlugin(const PluginTensorDesc* in, int nbInput, const PluginTensorDesc* out, int nbOutput) TRT_NOEXCEPT override;

		size_t getWorkspaceSize(int maxBatchSize) const TRT_NOEXCEPT override;
		
		int enqueue(int batchSize, const void* const* inputs, void* TRT_CONST_ENQUEUE* outputs, void* workspace, cudaStream_t stream) TRT_NOEXCEPT override;

		// Return true if output tensor is broadcast across a batch.
		bool isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const TRT_NOEXCEPT override;

		// Return true if plugin can use input that is broadcast across batch without replication.
		bool canBroadcastInputAcrossBatch(int inputIndex) const TRT_NOEXCEPT override;

		// IPluginV2IOExt Methods
		DataType getOutputDataType(int index, const DataType* inputTypes, int nbInputs) const TRT_NOEXCEPT override;

		void attachToContext(cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) TRT_NOEXCEPT override;

		void detachFromContext() TRT_NOEXCEPT override;

		const char* getPluginType() const TRT_NOEXCEPT override;

		const char* getPluginVersion() const TRT_NOEXCEPT override;

		int getNbOutputs() const TRT_NOEXCEPT override;

		int initialize() TRT_NOEXCEPT override;

		void terminate() TRT_NOEXCEPT override;

		size_t getSerializationSize() const TRT_NOEXCEPT override;

		void serialize(void* buffer) const TRT_NOEXCEPT override;

		void destroy() TRT_NOEXCEPT override;

		void setPluginNamespace(const char* pluginNamespace) TRT_NOEXCEPT override;

		const char* getPluginNamespace() const TRT_NOEXCEPT override;

	public:
		DCNv2Parameters mParam;
		const std::string mLayerName;
		std::string mNamespace;
		cublasHandle_t mCublas;
		cudnnHandle_t mCudnn;
		bool _initialized;
		
	}; // class DCNv2Plugin

	class DCNv2PluginCreator : public IPluginCreator
	{
	public:
		DCNv2PluginCreator();

		const char* getPluginName() const TRT_NOEXCEPT override;

		const char* getPluginVersion() const TRT_NOEXCEPT override;

		const PluginFieldCollection* getFieldNames() TRT_NOEXCEPT override;

		IPluginV2IOExt* createPlugin(const char* name, const PluginFieldCollection* fc) TRT_NOEXCEPT override;

		IPluginV2IOExt* deserializePlugin(const char* name, const void* serialData, size_t serialLength) TRT_NOEXCEPT override;

		void setPluginNamespace(const char* pluginNamespace) TRT_NOEXCEPT override;

		const char* getPluginNamespace() const TRT_NOEXCEPT override;

	private:
		static PluginFieldCollection mFC;
		static std::vector<PluginField> mPluginAttributes;

		std::string mNamespace;
	}; // class DCNv2PluginCreator


	REGISTER_TENSORRT_PLUGIN(DCNv2PluginCreator);
} //namespace nvinfer1

#endif // __TRT_DCNV2_PLUGIN_H__
