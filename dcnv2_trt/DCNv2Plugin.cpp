#include "DCNv2Plugin.h"

#ifndef CHECK_CUDA
#define CHECK_CUDA(callstr)\
    {\
        cudaError_t error_code = callstr;\
        if (error_code != cudaSuccess) {\
            std::cerr << "CUDA error " << error_code << " at " << __FILE__ << ":" << __LINE__;\
            assert(0);\
        }\
    }
#endif  // CHECK_CUDA

// Write values into buffer
template <typename T>
void write(char*& buffer, const T& val)
{
    std::memcpy(buffer, &val, sizeof(T));
    buffer += sizeof(T);
}

// Read values from buffer
template <typename T>
T read(const char*& buffer)
{
    T val{};
    std::memcpy(&val, buffer, sizeof(T));
    buffer += sizeof(T);
    return val;
}


namespace nvinfer1
{

static const char* DCNV2_NAME{"DCNv2_TRT"};
static const char* DCNV2_VERSION{"1"};

// Static class fields initialization
PluginFieldCollection DCNv2PluginCreator::mFC{};
std::vector<PluginField> DCNv2PluginCreator::mPluginAttributes;

DCNv2Plugin::DCNv2Plugin(const void* data, size_t length, const std::string& name)
	: mLayerName(name)
{
	const char* d = reinterpret_cast<const char*>(data);//, *a = d;

	for (int i = 0 ; i < DILATION_DIM ; i++)
		mParam.dilation.push_back(read<int>(d));
	for (int i = 0 ; i < PADDING_DIM ; i++)
		mParam.padding.push_back(read<int>(d));
	for (int i = 0 ; i < STRIDE_DIM ; i++)
		mParam.stride.push_back(read<int>(d));
	mParam.deformable_groups = read<int>(d);
	mParam.oc = read<int>(d);
	mParam.in_channel 	= read<int>(d);
	mParam.in_w 		= read<int>(d);
	mParam.in_h 		= read<int>(d);
	mParam.out_channel 	= read<int>(d);
	mParam.kernel_h 	= read<int>(d);
	mParam.kernel_w 	= read<int>(d);
	mParam.out_h 		= read<int>(d);
	mParam.out_w 		= read<int>(d);

	_initialized = false;
	mParam.mOne = nullptr;
	mParam.mColumn = nullptr;
	cublasCreate(&mCublas);
}

DCNv2Plugin::DCNv2Plugin(DCNv2Parameters param, const std::string& name)
	: mParam{param}, mLayerName(name)
{
	cublasCreate(&mCublas);
}

DCNv2Plugin::~DCNv2Plugin()
{terminate();
}

// IPluginV2IOExt Methods
IPluginV2IOExt* DCNv2Plugin::clone() const TRT_NOEXCEPT
{
	DCNv2Plugin* p = new DCNv2Plugin(mParam, mLayerName);
	p->setPluginNamespace(mNamespace.c_str());
	return p;
}

Dims DCNv2Plugin::getOutputDimensions(int index, const Dims* inputs, int nbInputDims) TRT_NOEXCEPT
{
	// Validate input dimention
	assert(nbInputDims == 5);
	Dims ret;
	ret.nbDims = 3;
	ret.d[0] = mParam.oc;
	ret.d[1] = inputs[0].d[1];
	ret.d[2] = inputs[0].d[2];

	Dims input_shape   = inputs[0];
	Dims weights_shape = inputs[3];

	mParam.in_channel 		= input_shape.d[0];
	mParam.in_w 			= input_shape.d[1];
	mParam.in_h 			= input_shape.d[2];
	mParam.out_channel 		= mParam.oc;
	mParam.kernel_h 		= weights_shape.d[1];
	mParam.kernel_w 		= weights_shape.d[2];
	mParam.out_h 			= mParam.in_h;
	mParam.out_w 			= mParam.in_w;

	return ret;
}

bool DCNv2Plugin::supportsFormatCombination(int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) const TRT_NOEXCEPT
{
	// Validate input arguments
	assert(nbInputs == 5);
	assert(nbOutputs == 1);

	const PluginTensorDesc& desc = inOut[pos];

	return ((desc.type == DataType::kFLOAT || desc.type == DataType::kHALF) && desc.format == TensorFormat::kLINEAR);
}

void DCNv2Plugin::configurePlugin(const PluginTensorDesc* in, int nbInput, const PluginTensorDesc* out, int nbOutput) TRT_NOEXCEPT
{
}

size_t DCNv2Plugin::getWorkspaceSize(int maxBatchSize) const TRT_NOEXCEPT
{
	return 0;
}

int DCNv2Plugin::enqueue(int batchSize, const void* const* inputs, void* TRT_CONST_ENQUEUE* outputs, void* workspace, cudaStream_t stream) TRT_NOEXCEPT
{
    const float* input  = static_cast<const float *>(inputs[0]);
	const float* offset = static_cast<const float *>(inputs[1]);
	const float* mask   = static_cast<const float *>(inputs[2]);
	const float* wights = static_cast<const float *>(inputs[3]);
	const float* bias   = static_cast<const float *>(inputs[4]);

    float * output = static_cast<float *>(outputs[0]);

// std::cout << 	" | in_channel: "	 	<< mParam.in_channel <<
// 				" | in_w: "		 		<< mParam.in_w <<
// 				" | in_h: "		 		<< mParam.in_h <<
// 				" | out_channel: "	 	<< mParam.out_channel <<
// 				" | kernel_h: "	 		<< mParam.kernel_h <<
// 				" | kernel_w: "	 		<< mParam.kernel_w <<
// 				" | out_h: "		 	<< mParam.out_h <<
// 				" | out_w: "		 	<< mParam.out_w << std::endl;


    const float alpha = 1.0f;
	const float beta = 0.0f;

    // Do Bias first:
    // M,N,K are dims of matrix A and B
    // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
    // (N x 1) (1 x M)
    int m_ = mParam.out_channel;
    int n_ = mParam.out_h * mParam.out_w;
    int k_ = 1;

    cublasSgemm(mCublas, CUBLAS_OP_T, CUBLAS_OP_N, n_, m_, k_, &alpha,
                mParam.mOne, k_,
                bias, k_, &beta,
                output, n_);

    modulated_deformable_im2col_cuda(stream, input, offset, mask,
                                    1, mParam.in_channel, mParam.in_h, mParam.in_w,
                                    mParam.out_h, mParam.out_w, mParam.kernel_h, mParam.kernel_w,
                                    mParam.padding[0], mParam.padding[1], mParam.stride[0], mParam.stride[1],
                                    mParam.dilation[0], mParam.dilation[1], mParam.deformable_groups, mParam.mColumn); 

    //(k * m)  x  (m * n)
    // Y = WC
    int m = mParam.out_channel;
    int n = mParam.out_h * mParam.out_w;
    int k = mParam.in_channel * mParam.kernel_h * mParam.kernel_w;
    cublasSgemm(mCublas, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha,
                mParam.mColumn, n,
                wights, k, &alpha,
                output, n);
	
	return 0;
}

DataType DCNv2Plugin::getOutputDataType(int index, const DataType* inputTypes, int nbInputs) const TRT_NOEXCEPT
{
	assert(inputTypes[0] == DataType::kFLOAT || inputTypes[0] == DataType::kHALF);
	return inputTypes[0];
}

// Return true if output tensor is broadcast across a batch.
bool DCNv2Plugin::isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const TRT_NOEXCEPT
{
	return false;
}

// Return true if plugin can use input that is broadcast across batch without replication.
bool DCNv2Plugin::canBroadcastInputAcrossBatch(int inputIndex) const TRT_NOEXCEPT
{
	return false;
}

// Attach the plugin object to an execution context and grant the plugin
// the access to some context resource
void DCNv2Plugin::attachToContext(cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) TRT_NOEXCEPT
{
}

// Detach the plugin object from its execution context
void DCNv2Plugin::detachFromContext() TRT_NOEXCEPT {}

// IPluginV2 Methods
const char* DCNv2Plugin::getPluginType() const TRT_NOEXCEPT
{
	return DCNV2_NAME;
}

const char* DCNv2Plugin::getPluginVersion() const TRT_NOEXCEPT
{
	return DCNV2_VERSION;
}

int DCNv2Plugin::getNbOutputs() const TRT_NOEXCEPT
{
	return 1;
}

int DCNv2Plugin::initialize() TRT_NOEXCEPT
{	
	if(_initialized) return 0;
    size_t oneSize = mParam.out_h * mParam.out_w;

    // std::vector<float> one_((int)oneSize, 1.0f);
    float *ones_cpu = new float[oneSize];
    for (size_t i = 0; i < oneSize; i++) {
        ones_cpu[i] = 1.0f;
    }

    CHECK_CUDA(cudaMalloc((void**)&mParam.mOne, oneSize * sizeof(float)));
	CHECK_CUDA(cudaMemcpy(mParam.mOne, ones_cpu, oneSize * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMalloc((void**)&mParam.mColumn, mParam.in_channel * mParam.kernel_h * mParam.kernel_w * oneSize * sizeof(float)));

    delete[] ones_cpu;
	_initialized = true;
	return 0;
}

void DCNv2Plugin::terminate() TRT_NOEXCEPT
{
	if (!_initialized)  return;
	cudaFree(mParam.mOne);
	cudaFree(mParam.mColumn);
    cublasDestroy(mCublas);
    _initialized = false;
}

size_t DCNv2Plugin::getSerializationSize() const TRT_NOEXCEPT
{
	return DILATION_DIM * sizeof(int)	// dilation
		+ PADDING_DIM * sizeof(int)	// padding
		+ STRIDE_DIM * sizeof(int)	// stride
		+ 1 * sizeof(int)				// deformable
		+ 9 * sizeof(int);
}

void DCNv2Plugin::serialize(void* buffer) const TRT_NOEXCEPT
{
	char* d = reinterpret_cast<char*>(buffer);//, *a = d;
	for (int i = 0 ; i < DILATION_DIM ; i++)
		write(d, mParam.dilation[i]);
	for (int i = 0 ; i < PADDING_DIM ; i++)
		write(d, mParam.padding[i]);
	for (int i = 0 ; i < STRIDE_DIM ; i++)
		write(d, mParam.stride[i]);
	write(d, mParam.deformable_groups);

	write(d, mParam.oc);
	write(d, mParam.in_channel);
	write(d, mParam.in_w);
	write(d, mParam.in_h);
	write(d, mParam.out_channel);
	write(d, mParam.kernel_h);
	write(d, mParam.kernel_w);
	write(d, mParam.out_h);
	write(d, mParam.out_w);
}

void DCNv2Plugin::destroy() TRT_NOEXCEPT
{
	delete this;
}

void DCNv2Plugin::setPluginNamespace(const char* pluginNamespace) TRT_NOEXCEPT
{
	mNamespace = pluginNamespace;
}

const char* DCNv2Plugin::getPluginNamespace() const TRT_NOEXCEPT
{
	return mNamespace.c_str();
}



DCNv2PluginCreator::DCNv2PluginCreator()
{
	mPluginAttributes.emplace_back(PluginField("dilation", nullptr,
		PluginFieldType::kINT32, 2));
	mPluginAttributes.emplace_back(PluginField("padding", nullptr,
		PluginFieldType::kINT32, 2));
	mPluginAttributes.emplace_back(PluginField("stride", nullptr,
		PluginFieldType::kINT32, 2));
	mPluginAttributes.emplace_back(PluginField("deformable_groups", nullptr,
		PluginFieldType::kINT32, 1));
	mPluginAttributes.emplace_back(PluginField("oc", nullptr,
		PluginFieldType::kINT32, 1));
	
	mFC.nbFields = mPluginAttributes.size();
	mFC.fields = mPluginAttributes.data();
}

const char* DCNv2PluginCreator::getPluginName() const TRT_NOEXCEPT
{
	return DCNV2_NAME;
}

const char* DCNv2PluginCreator::getPluginVersion() const TRT_NOEXCEPT
{
	return DCNV2_VERSION;
}

const PluginFieldCollection* DCNv2PluginCreator::getFieldNames() TRT_NOEXCEPT
{
	return &mFC;
}

IPluginV2IOExt* DCNv2PluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) TRT_NOEXCEPT
{
	std::vector<int> dilation;
	std::vector<int> padding;
	std::vector<int> stride;
	int deformable_groups = 1;
	int oc = 1;
	const PluginField* fields = fc->fields;

	for (int i = 0 ; i < fc->nbFields ; i++)
	{
		const char* attrName = fields[i].name;
		if (!strcmp(attrName, "dilation"))
		{
			assert(fields[i].type == PluginFieldType::kINT32);
			int size = fields[i].length;
			dilation.reserve(size);
			const auto* d = static_cast<const int*>(fields[i].data);
			for (int j = 0 ; j < size ; j++)
			{
				dilation.push_back(*d);
				d++;
			}
		}
		else if (!strcmp(attrName, "padding"))
		{
			assert(fields[i].type == PluginFieldType::kINT32);
			int size = fields[i].length;
			padding.reserve(size);
			const auto* p = static_cast<const int*>(fields[i].data);
			for (int j = 0 ; j < size ; j++)
			{
				padding.push_back(*p);
				p++;
			}
		}
		else if (!strcmp(attrName, "stride"))
		{
			assert(fields[i].type == PluginFieldType::kINT32);
			int size = fields[i].length;
			stride.reserve(size);
			const auto* s = static_cast<const int*>(fields[i].data);
			for (int j = 0 ; j < size ; j++)
			{
				stride.push_back(*s);
				s++;
			}
		}
        else if (!strcmp(attrName, "deformable_groups"))
        {
            assert(fields[i].type == PluginFieldType::kINT32);
            deformable_groups = *(static_cast<const int*>(fields[i].data));
        }
        else if (!strcmp(attrName, "oc"))
        {
            assert(fields[i].type == PluginFieldType::kINT32);
            oc = *(static_cast<const int*>(fields[i].data));
        }
	}

	DCNv2Parameters dcnv2Params;
	dcnv2Params.dilation = dilation;
	dcnv2Params.padding = padding;
	dcnv2Params.stride = stride;
	dcnv2Params.deformable_groups = deformable_groups;
	dcnv2Params.oc = oc;

	DCNv2Plugin* p = new DCNv2Plugin(dcnv2Params, name);
	p->setPluginNamespace(mNamespace.c_str());
	return p;
}

IPluginV2IOExt* DCNv2PluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength) TRT_NOEXCEPT
{
	// This object will be deleted when the network is destroyed, which will
	// call DCNv2Plugin::destroy()
	return new DCNv2Plugin(serialData, serialLength, std::string(name));
}

void DCNv2PluginCreator::setPluginNamespace(const char* pluginNamespace) TRT_NOEXCEPT
{
	mNamespace = pluginNamespace;
}

const char* DCNv2PluginCreator::getPluginNamespace() const TRT_NOEXCEPT
{
	return mNamespace.c_str();
}

} // namespace nvinfer1
