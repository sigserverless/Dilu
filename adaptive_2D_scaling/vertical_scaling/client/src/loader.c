#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <regex.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "../include/cuda-helper.h"
#include "../include/hijack.h"
#include "../include/nvml-helper.h"

entry_t cuda_library_entry[] = {
    {.name = "cuInit"},
    {.name = "cuDeviceGet"},
    {.name = "cuDeviceGetCount"},
    {.name = "cuDeviceGetName"},
    {.name = "cuDeviceTotalMem_v2"},
    {.name = "cuDeviceGetAttribute"},
    {.name = "cuDeviceGetP2PAttribute"},
    {.name = "cuDriverGetVersion"},
    {.name = "cuDeviceGetByPCIBusId"},
    {.name = "cuDeviceGetPCIBusId"},
    {.name = "cuDevicePrimaryCtxRetain"},
    {.name = "cuDevicePrimaryCtxRelease"},
    {.name = "cuDevicePrimaryCtxSetFlags"},
    {.name = "cuDevicePrimaryCtxGetState"},
    {.name = "cuDevicePrimaryCtxReset"},
    {.name = "cuCtxCreate_v2"},
    {.name = "cuCtxGetFlags"},
    {.name = "cuCtxSetCurrent"},
    {.name = "cuCtxGetCurrent"},
    {.name = "cuCtxDetach"},
    {.name = "cuCtxGetApiVersion"},
    {.name = "cuCtxGetDevice"},
    {.name = "cuCtxGetLimit"},
    {.name = "cuCtxSetLimit"},
    {.name = "cuCtxGetCacheConfig"},
    {.name = "cuCtxSetCacheConfig"},
    {.name = "cuCtxGetSharedMemConfig"},
    {.name = "cuCtxGetStreamPriorityRange"},
    {.name = "cuCtxSetSharedMemConfig"},
    {.name = "cuCtxSynchronize"},
    {.name = "cuModuleLoad"},
    {.name = "cuModuleLoadData"},
    {.name = "cuModuleLoadFatBinary"},
    {.name = "cuModuleUnload"},
    {.name = "cuModuleGetFunction"},
    {.name = "cuModuleGetGlobal_v2"},
    {.name = "cuModuleGetTexRef"},
    {.name = "cuModuleGetSurfRef"},
    {.name = "cuLinkCreate"},
    {.name = "cuLinkAddData"},
    {.name = "cuLinkAddFile"},
    {.name = "cuLinkComplete"},
    {.name = "cuLinkDestroy"},
    {.name = "cuMemGetInfo_v2"},
    {.name = "cuMemAllocManaged"},
    {.name = "cuMemAlloc_v2"},
    {.name = "cuMemAllocPitch_v2"},
    {.name = "cuMemFree_v2"},
    {.name = "cuMemGetAddressRange_v2"},
    {.name = "cuMemFreeHost"},
    {.name = "cuMemHostAlloc"},
    {.name = "cuMemHostGetDevicePointer_v2"},
    {.name = "cuMemHostGetFlags"},
    {.name = "cuMemHostRegister_v2"},
    {.name = "cuMemHostUnregister"},
    {.name = "cuPointerGetAttribute"},
    {.name = "cuPointerGetAttributes"},
    {.name = "cuMemcpy"},
    {.name = "cuMemcpy_ptds"},
    {.name = "cuMemcpyAsync"},
    {.name = "cuMemcpyAsync_ptsz"},
    {.name = "cuMemcpyPeer"},
    {.name = "cuMemcpyPeer_ptds"},
    {.name = "cuMemcpyPeerAsync"},
    {.name = "cuMemcpyPeerAsync_ptsz"},
    {.name = "cuMemcpyHtoD_v2"},
    {.name = "cuMemcpyHtoD_v2_ptds"},
    {.name = "cuMemcpyHtoDAsync_v2"},
    {.name = "cuMemcpyHtoDAsync_v2_ptsz"},
    {.name = "cuMemcpyDtoH_v2"},
    {.name = "cuMemcpyDtoH_v2_ptds"},
    {.name = "cuMemcpyDtoHAsync_v2"},
    {.name = "cuMemcpyDtoHAsync_v2_ptsz"},
    {.name = "cuMemcpyDtoD_v2"},
    {.name = "cuMemcpyDtoD_v2_ptds"},
    {.name = "cuMemcpyDtoDAsync_v2"},
    {.name = "cuMemcpyDtoDAsync_v2_ptsz"},
    {.name = "cuMemcpy2DUnaligned_v2"},
    {.name = "cuMemcpy2DUnaligned_v2_ptds"},
    {.name = "cuMemcpy2DAsync_v2"},
    {.name = "cuMemcpy2DAsync_v2_ptsz"},
    {.name = "cuMemcpy3D_v2"},
    {.name = "cuMemcpy3D_v2_ptds"},
    {.name = "cuMemcpy3DAsync_v2"},
    {.name = "cuMemcpy3DAsync_v2_ptsz"},
    {.name = "cuMemcpy3DPeer"},
    {.name = "cuMemcpy3DPeer_ptds"},
    {.name = "cuMemcpy3DPeerAsync"},
    {.name = "cuMemcpy3DPeerAsync_ptsz"},
    {.name = "cuMemsetD8_v2"},
    {.name = "cuMemsetD8_v2_ptds"},
    {.name = "cuMemsetD8Async"},
    {.name = "cuMemsetD8Async_ptsz"},
    {.name = "cuMemsetD2D8_v2"},
    {.name = "cuMemsetD2D8_v2_ptds"},
    {.name = "cuMemsetD2D8Async"},
    {.name = "cuMemsetD2D8Async_ptsz"},
    {.name = "cuFuncSetCacheConfig"},
    {.name = "cuFuncSetSharedMemConfig"},
    {.name = "cuFuncGetAttribute"},
    {.name = "cuArrayCreate_v2"},
    {.name = "cuArrayGetDescriptor_v2"},
    {.name = "cuArray3DCreate_v2"},
    {.name = "cuArray3DGetDescriptor_v2"},
    {.name = "cuArrayDestroy"},
    {.name = "cuMipmappedArrayCreate"},
    {.name = "cuMipmappedArrayGetLevel"},
    {.name = "cuMipmappedArrayDestroy"},
    {.name = "cuTexRefCreate"},
    {.name = "cuTexRefDestroy"},
    {.name = "cuTexRefSetArray"},
    {.name = "cuTexRefSetMipmappedArray"},
    {.name = "cuTexRefSetAddress_v2"},
    {.name = "cuTexRefSetAddress2D_v3"},
    {.name = "cuTexRefSetFormat"},
    {.name = "cuTexRefSetAddressMode"},
    {.name = "cuTexRefSetFilterMode"},
    {.name = "cuTexRefSetMipmapFilterMode"},
    {.name = "cuTexRefSetMipmapLevelBias"},
    {.name = "cuTexRefSetMipmapLevelClamp"},
    {.name = "cuTexRefSetMaxAnisotropy"},
    {.name = "cuTexRefSetFlags"},
    {.name = "cuTexRefSetBorderColor"},
    {.name = "cuTexRefGetBorderColor"},
    {.name = "cuSurfRefSetArray"},
    {.name = "cuTexObjectCreate"},
    {.name = "cuTexObjectDestroy"},
    {.name = "cuTexObjectGetResourceDesc"},
    {.name = "cuTexObjectGetTextureDesc"},
    {.name = "cuTexObjectGetResourceViewDesc"},
    {.name = "cuSurfObjectCreate"},
    {.name = "cuSurfObjectDestroy"},
    {.name = "cuSurfObjectGetResourceDesc"},
    {.name = "cuLaunchKernel"},
    {.name = "cuLaunchKernel_ptsz"},
    {.name = "cuEventCreate"},
    {.name = "cuEventRecord"},
    {.name = "cuEventRecord_ptsz"},
    {.name = "cuEventQuery"},
    {.name = "cuEventSynchronize"},
    {.name = "cuEventDestroy_v2"},
    {.name = "cuEventElapsedTime"},
    {.name = "cuStreamWaitValue32"},
    {.name = "cuStreamWaitValue32_ptsz"},
    {.name = "cuStreamWriteValue32"},
    {.name = "cuStreamWriteValue32_ptsz"},
    {.name = "cuStreamBatchMemOp"},
    {.name = "cuStreamBatchMemOp_ptsz"},
    {.name = "cuStreamCreate"},
    {.name = "cuStreamCreateWithPriority"},
    {.name = "cuStreamGetPriority"},
    {.name = "cuStreamGetPriority_ptsz"},
    {.name = "cuStreamGetFlags"},
    {.name = "cuStreamGetFlags_ptsz"},
    {.name = "cuStreamDestroy_v2"},
    {.name = "cuStreamWaitEvent"},
    {.name = "cuStreamWaitEvent_ptsz"},
    {.name = "cuStreamAddCallback"},
    {.name = "cuStreamAddCallback_ptsz"},
    {.name = "cuStreamSynchronize"},
    {.name = "cuStreamSynchronize_ptsz"},
    {.name = "cuStreamQuery"},
    {.name = "cuStreamQuery_ptsz"},
    {.name = "cuStreamAttachMemAsync"},
    {.name = "cuStreamAttachMemAsync_ptsz"},
    {.name = "cuDeviceCanAccessPeer"},
    {.name = "cuCtxEnablePeerAccess"},
    {.name = "cuCtxDisablePeerAccess"},
    // {.name = "cuGetProcAddress"},
    {.name = "cuIpcGetEventHandle"},
    {.name = "cuIpcOpenEventHandle"},
    {.name = "cuIpcGetMemHandle"},
    {.name = "cuIpcOpenMemHandle"},
    {.name = "cuIpcCloseMemHandle"},
    {.name = "cuGLCtxCreate_v2"},
    {.name = "cuGLInit"},
    {.name = "cuGLGetDevices"},
    {.name = "cuGLRegisterBufferObject"},
    {.name = "cuGLMapBufferObject_v2"},
    {.name = "cuGLMapBufferObject_v2_ptds"},
    {.name = "cuGLMapBufferObjectAsync_v2"},
    {.name = "cuGLMapBufferObjectAsync_v2_ptsz"},
    {.name = "cuGLUnmapBufferObject"},
    {.name = "cuGLUnmapBufferObjectAsync"},
    {.name = "cuGLUnregisterBufferObject"},
    {.name = "cuGLSetBufferObjectMapFlags"},
    {.name = "cuGraphicsGLRegisterImage"},
    {.name = "cuGraphicsGLRegisterBuffer"},
    {.name = "cuGraphicsUnregisterResource"},
    {.name = "cuGraphicsMapResources"},
    {.name = "cuGraphicsMapResources_ptsz"},
    {.name = "cuGraphicsUnmapResources"},
    {.name = "cuGraphicsUnmapResources_ptsz"},
    {.name = "cuGraphicsResourceSetMapFlags_v2"},
    {.name = "cuGraphicsSubResourceGetMappedArray"},
    {.name = "cuGraphicsResourceGetMappedMipmappedArray"},
    {.name = "cuGraphicsResourceGetMappedPointer_v2"},
    {.name = "cuProfilerInitialize"},
    {.name = "cuProfilerStart"},
    {.name = "cuProfilerStop"},
    {.name = "cuVDPAUGetDevice"},
    {.name = "cuVDPAUCtxCreate_v2"},
    {.name = "cuGraphicsVDPAURegisterVideoSurface"},
    {.name = "cuGraphicsVDPAURegisterOutputSurface"},
    {.name = "cuGetExportTable"},
    {.name = "cuOccupancyMaxActiveBlocksPerMultiprocessor"},
    {.name = "cuMemAdvise"},
    {.name = "cuMemPrefetchAsync"},
    {.name = "cuMemPrefetchAsync_ptsz"},
    {.name = "cuMemRangeGetAttribute"},
    {.name = "cuMemRangeGetAttributes"},
    {.name = "cuGetErrorString"},
    {.name = "cuGetErrorName"},
    {.name = "cuArray3DCreate"},
    {.name = "cuArray3DGetDescriptor"},
    {.name = "cuArrayCreate"},
    {.name = "cuArrayGetDescriptor"},
    {.name = "cuCtxAttach"},
    {.name = "cuCtxCreate"},
    {.name = "cuCtxDestroy"},
    {.name = "cuCtxDestroy_v2"},
    {.name = "cuCtxPopCurrent"},
    {.name = "cuCtxPopCurrent_v2"},
    {.name = "cuCtxPushCurrent"},
    {.name = "cuCtxPushCurrent_v2"},
    {.name = "cudbgApiAttach"},
    {.name = "cudbgApiDetach"},
    {.name = "cudbgApiInit"},
    {.name = "cudbgGetAPI"},
    {.name = "cudbgGetAPIVersion"},
    {.name = "cudbgMain"},
    {.name = "cudbgReportDriverApiError"},
    {.name = "cudbgReportDriverInternalError"},
    {.name = "cuDeviceComputeCapability"},
    {.name = "cuDeviceGetProperties"},
    {.name = "cuDeviceTotalMem"},
    {.name = "cuEGLInit"},
    {.name = "cuEGLStreamConsumerAcquireFrame"},
    {.name = "cuEGLStreamConsumerConnect"},
    {.name = "cuEGLStreamConsumerConnectWithFlags"},
    {.name = "cuEGLStreamConsumerDisconnect"},
    {.name = "cuEGLStreamConsumerReleaseFrame"},
    {.name = "cuEGLStreamProducerConnect"},
    {.name = "cuEGLStreamProducerDisconnect"},
    {.name = "cuEGLStreamProducerPresentFrame"},
    {.name = "cuEGLStreamProducerReturnFrame"},
    {.name = "cuEventDestroy"},
    {.name = "cuFuncSetAttribute"},
    {.name = "cuFuncSetBlockShape"},
    {.name = "cuFuncSetSharedSize"},
    {.name = "cuGLCtxCreate"},
    {.name = "cuGLGetDevices_v2"},
    {.name = "cuGLMapBufferObject"},
    {.name = "cuGLMapBufferObjectAsync"},
    {.name = "cuGraphicsEGLRegisterImage"},
    {.name = "cuGraphicsResourceGetMappedEglFrame"},
    {.name = "cuGraphicsResourceGetMappedPointer"},
    {.name = "cuGraphicsResourceSetMapFlags"},
    {.name = "cuLaunch"},
    {.name = "cuLaunchCooperativeKernel"},
    {.name = "cuLaunchCooperativeKernelMultiDevice"},
    {.name = "cuLaunchCooperativeKernel_ptsz"},
    {.name = "cuLaunchGrid"},
    {.name = "cuLaunchGridAsync"},
    {.name = "cuLinkAddData_v2"},
    {.name = "cuLinkAddFile_v2"},
    {.name = "cuLinkCreate_v2"},
    {.name = "cuMemAlloc"},
    {.name = "cuMemAllocHost"},
    {.name = "cuMemAllocHost_v2"},
    {.name = "cuMemAllocPitch"},
    {.name = "cuMemcpy2D"},
    {.name = "cuMemcpy2DAsync"},
    {.name = "cuMemcpy2DUnaligned"},
    {.name = "cuMemcpy2D_v2"},
    {.name = "cuMemcpy2D_v2_ptds"},
    {.name = "cuMemcpy3D"},
    {.name = "cuMemcpy3DAsync"},
    {.name = "cuMemcpyAtoA"},
    {.name = "cuMemcpyAtoA_v2"},
    {.name = "cuMemcpyAtoA_v2_ptds"},
    {.name = "cuMemcpyAtoD"},
    {.name = "cuMemcpyAtoD_v2"},
    {.name = "cuMemcpyAtoD_v2_ptds"},
    {.name = "cuMemcpyAtoH"},
    {.name = "cuMemcpyAtoHAsync"},
    {.name = "cuMemcpyAtoHAsync_v2"},
    {.name = "cuMemcpyAtoHAsync_v2_ptsz"},
    {.name = "cuMemcpyAtoH_v2"},
    {.name = "cuMemcpyAtoH_v2_ptds"},
    {.name = "cuMemcpyDtoA"},
    {.name = "cuMemcpyDtoA_v2"},
    {.name = "cuMemcpyDtoA_v2_ptds"},
    {.name = "cuMemcpyDtoD"},
    {.name = "cuMemcpyDtoDAsync"},
    {.name = "cuMemcpyDtoH"},
    {.name = "cuMemcpyDtoHAsync"},
    {.name = "cuMemcpyHtoA"},
    {.name = "cuMemcpyHtoAAsync"},
    {.name = "cuMemcpyHtoAAsync_v2"},
    {.name = "cuMemcpyHtoAAsync_v2_ptsz"},
    {.name = "cuMemcpyHtoA_v2"},
    {.name = "cuMemcpyHtoA_v2_ptds"},
    {.name = "cuMemcpyHtoD"},
    {.name = "cuMemcpyHtoDAsync"},
    {.name = "cuMemFree"},
    {.name = "cuMemGetAddressRange"},
    {.name = "cuMemGetInfo"},
    {.name = "cuMemHostGetDevicePointer"},
    {.name = "cuMemHostRegister"},
    {.name = "cuMemsetD16"},
    {.name = "cuMemsetD16Async"},
    {.name = "cuMemsetD16Async_ptsz"},
    {.name = "cuMemsetD16_v2"},
    {.name = "cuMemsetD16_v2_ptds"},
    {.name = "cuMemsetD2D16"},
    {.name = "cuMemsetD2D16Async"},
    {.name = "cuMemsetD2D16Async_ptsz"},
    {.name = "cuMemsetD2D16_v2"},
    {.name = "cuMemsetD2D16_v2_ptds"},
    {.name = "cuMemsetD2D32"},
    {.name = "cuMemsetD2D32Async"},
    {.name = "cuMemsetD2D32Async_ptsz"},
    {.name = "cuMemsetD2D32_v2"},
    {.name = "cuMemsetD2D32_v2_ptds"},
    {.name = "cuMemsetD2D8"},
    {.name = "cuMemsetD32"},
    {.name = "cuMemsetD32Async"},
    {.name = "cuMemsetD32Async_ptsz"},
    {.name = "cuMemsetD32_v2"},
    {.name = "cuMemsetD32_v2_ptds"},
    {.name = "cuMemsetD8"},
    {.name = "cuModuleGetGlobal"},
    {.name = "cuModuleLoadDataEx"},
    {.name = "cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags"},
    {.name = "cuOccupancyMaxPotentialBlockSize"},
    {.name = "cuOccupancyMaxPotentialBlockSizeWithFlags"},
    {.name = "cuParamSetf"},
    {.name = "cuParamSeti"},
    {.name = "cuParamSetSize"},
    {.name = "cuParamSetTexRef"},
    {.name = "cuParamSetv"},
    {.name = "cuPointerSetAttribute"},
    {.name = "cuStreamDestroy"},
    {.name = "cuStreamWaitValue64"},
    {.name = "cuStreamWaitValue64_ptsz"},
    {.name = "cuStreamWriteValue64"},
    {.name = "cuStreamWriteValue64_ptsz"},
    {.name = "cuSurfRefGetArray"},
    {.name = "cuTexRefGetAddress"},
    {.name = "cuTexRefGetAddressMode"},
    {.name = "cuTexRefGetAddress_v2"},
    {.name = "cuTexRefGetArray"},
    {.name = "cuTexRefGetFilterMode"},
    {.name = "cuTexRefGetFlags"},
    {.name = "cuTexRefGetFormat"},
    {.name = "cuTexRefGetMaxAnisotropy"},
    {.name = "cuTexRefGetMipmapFilterMode"},
    {.name = "cuTexRefGetMipmapLevelBias"},
    {.name = "cuTexRefGetMipmapLevelClamp"},
    {.name = "cuTexRefGetMipmappedArray"},
    {.name = "cuTexRefSetAddress"},
    {.name = "cuTexRefSetAddress2D"},
    {.name = "cuTexRefSetAddress2D_v2"},
    {.name = "cuVDPAUCtxCreate"},
    {.name = "cuEGLApiInit"},
    {.name = "cuDestroyExternalMemory"},
    {.name = "cuDestroyExternalSemaphore"},
    {.name = "cuDeviceGetUuid"},
    {.name = "cuExternalMemoryGetMappedBuffer"},
    {.name = "cuExternalMemoryGetMappedMipmappedArray"},
    {.name = "cuGraphAddChildGraphNode"},
    {.name = "cuGraphAddDependencies"},
    {.name = "cuGraphAddEmptyNode"},
    {.name = "cuGraphAddHostNode"},
    {.name = "cuGraphAddKernelNode"},
    {.name = "cuGraphAddMemcpyNode"},
    {.name = "cuGraphAddMemsetNode"},
    {.name = "cuGraphChildGraphNodeGetGraph"},
    {.name = "cuGraphClone"},
    {.name = "cuGraphCreate"},
    {.name = "cuGraphDestroy"},
    {.name = "cuGraphDestroyNode"},
    {.name = "cuGraphExecDestroy"},
    {.name = "cuGraphGetEdges"},
    {.name = "cuGraphGetNodes"},
    {.name = "cuGraphGetRootNodes"},
    {.name = "cuGraphHostNodeGetParams"},
    {.name = "cuGraphHostNodeSetParams"},
    {.name = "cuGraphInstantiate"},
    {.name = "cuGraphKernelNodeGetParams"},
    {.name = "cuGraphKernelNodeSetParams"},
    {.name = "cuGraphLaunch"},
    {.name = "cuGraphLaunch_ptsz"},
    {.name = "cuGraphMemcpyNodeGetParams"},
    {.name = "cuGraphMemcpyNodeSetParams"},
    {.name = "cuGraphMemsetNodeGetParams"},
    {.name = "cuGraphMemsetNodeSetParams"},
    {.name = "cuGraphNodeFindInClone"},
    {.name = "cuGraphNodeGetDependencies"},
    {.name = "cuGraphNodeGetDependentNodes"},
    {.name = "cuGraphNodeGetType"},
    {.name = "cuGraphRemoveDependencies"},
    {.name = "cuImportExternalMemory"},
    {.name = "cuImportExternalSemaphore"},
    {.name = "cuLaunchHostFunc"},
    {.name = "cuLaunchHostFunc_ptsz"},
    {.name = "cuSignalExternalSemaphoresAsync"},
    {.name = "cuSignalExternalSemaphoresAsync_ptsz"},
    {.name = "cuStreamBeginCapture"},
    {.name = "cuStreamBeginCapture_ptsz"},
    {.name = "cuStreamEndCapture"},
    {.name = "cuStreamEndCapture_ptsz"},
    {.name = "cuStreamGetCtx"},
    {.name = "cuStreamGetCtx_ptsz"},
    {.name = "cuStreamIsCapturing"},
    {.name = "cuStreamIsCapturing_ptsz"},
    {.name = "cuWaitExternalSemaphoresAsync"},
    {.name = "cuWaitExternalSemaphoresAsync_ptsz"},
    {.name = "cuGraphExecKernelNodeSetParams"},
    {.name = "cuStreamBeginCapture_v2"},
    {.name = "cuStreamBeginCapture_v2_ptsz"},
    {.name = "cuStreamGetCaptureInfo"},
    {.name = "cuStreamGetCaptureInfo_ptsz"},
    {.name = "cuThreadExchangeStreamCaptureMode"},
    {.name = "cuDeviceGetNvSciSyncAttributes"},
    {.name = "cuGraphExecHostNodeSetParams"},
    {.name = "cuGraphExecMemcpyNodeSetParams"},
    {.name = "cuGraphExecMemsetNodeSetParams"},
    {.name = "cuGraphExecUpdate"},
    {.name = "cuMemAddressFree"},
    {.name = "cuMemAddressReserve"},
    {.name = "cuMemCreate"},
    {.name = "cuMemExportToShareableHandle"},
    {.name = "cuMemGetAccess"},
    {.name = "cuMemGetAllocationGranularity"},
    {.name = "cuMemGetAllocationPropertiesFromHandle"},
    {.name = "cuMemImportFromShareableHandle"},
    {.name = "cuMemMap"},
    {.name = "cuMemRelease"},
    {.name = "cuMemSetAccess"},
    {.name = "cuMemUnmap"},
    {.name = "cuCtxResetPersistingL2Cache"},
    {.name = "cuDevicePrimaryCtxRelease_v2"},
    {.name = "cuDevicePrimaryCtxReset_v2"},
    {.name = "cuDevicePrimaryCtxSetFlags_v2"},
    {.name = "cuFuncGetModule"},
    {.name = "cuGraphInstantiate_v2"},
    {.name = "cuGraphKernelNodeCopyAttributes"},
    {.name = "cuGraphKernelNodeGetAttribute"},
    {.name = "cuGraphKernelNodeSetAttribute"},
    {.name = "cuMemRetainAllocationHandle"},
    {.name = "cuOccupancyAvailableDynamicSMemPerBlock"},
    {.name = "cuStreamCopyAttributes"},
    {.name = "cuStreamCopyAttributes_ptsz"},
    {.name = "cuStreamGetAttribute"},
    {.name = "cuStreamGetAttribute_ptsz"},
    {.name = "cuStreamSetAttribute"},
    {.name = "cuStreamSetAttribute_ptsz"},
};

entry_t nvml_library_entry[] = {
    {.name = "nvmlInit"},
    {.name = "nvmlShutdown"},
    {.name = "nvmlErrorString"},
    {.name = "nvmlDeviceGetHandleByIndex"},
    {.name = "nvmlDeviceGetComputeRunningProcesses"},
    {.name = "nvmlDeviceGetPciInfo"},
    {.name = "nvmlDeviceGetProcessUtilization"},
    {.name = "nvmlDeviceGetCount"},
    {.name = "nvmlDeviceClearAccountingPids"},
    {.name = "nvmlDeviceClearCpuAffinity"},
    {.name = "nvmlDeviceClearEccErrorCounts"},
    {.name = "nvmlDeviceDiscoverGpus"},
    {.name = "nvmlDeviceFreezeNvLinkUtilizationCounter"},
    {.name = "nvmlDeviceGetAccountingBufferSize"},
    {.name = "nvmlDeviceGetAccountingMode"},
    {.name = "nvmlDeviceGetAccountingPids"},
    {.name = "nvmlDeviceGetAccountingStats"},
    {.name = "nvmlDeviceGetActiveVgpus"},
    {.name = "nvmlDeviceGetAPIRestriction"},
    {.name = "nvmlDeviceGetApplicationsClock"},
    {.name = "nvmlDeviceGetAutoBoostedClocksEnabled"},
    {.name = "nvmlDeviceGetBAR1MemoryInfo"},
    {.name = "nvmlDeviceGetBoardId"},
    {.name = "nvmlDeviceGetBoardPartNumber"},
    {.name = "nvmlDeviceGetBrand"},
    {.name = "nvmlDeviceGetBridgeChipInfo"},
    {.name = "nvmlDeviceGetClock"},
    {.name = "nvmlDeviceGetClockInfo"},
    {.name = "nvmlDeviceGetComputeMode"},
    {.name = "nvmlDeviceGetCount_v2"},
    {.name = "nvmlDeviceGetCpuAffinity"},
    {.name = "nvmlDeviceGetCreatableVgpus"},
    {.name = "nvmlDeviceGetCudaComputeCapability"},
    {.name = "nvmlDeviceGetCurrentClocksThrottleReasons"},
    {.name = "nvmlDeviceGetCurrPcieLinkGeneration"},
    {.name = "nvmlDeviceGetCurrPcieLinkWidth"},
    {.name = "nvmlDeviceGetDecoderUtilization"},
    {.name = "nvmlDeviceGetDefaultApplicationsClock"},
    {.name = "nvmlDeviceGetDetailedEccErrors"},
    {.name = "nvmlDeviceGetDisplayActive"},
    {.name = "nvmlDeviceGetDisplayMode"},
    {.name = "nvmlDeviceGetDriverModel"},
    {.name = "nvmlDeviceGetEccMode"},
    {.name = "nvmlDeviceGetEncoderCapacity"},
    {.name = "nvmlDeviceGetEncoderSessions"},
    {.name = "nvmlDeviceGetEncoderStats"},
    {.name = "nvmlDeviceGetEncoderUtilization"},
    {.name = "nvmlDeviceGetEnforcedPowerLimit"},
    {.name = "nvmlDeviceGetFanSpeed"},
    {.name = "nvmlDeviceGetFanSpeed_v2"},
    {.name = "nvmlDeviceGetFieldValues"},
    {.name = "nvmlDeviceGetGpuOperationMode"},
    {.name = "nvmlDeviceGetGraphicsRunningProcesses"},
    {.name = "nvmlDeviceGetGridLicensableFeatures"},
    {.name = "nvmlDeviceGetHandleByIndex_v2"},
    {.name = "nvmlDeviceGetHandleByPciBusId"},
    {.name = "nvmlDeviceGetHandleByPciBusId_v2"},
    {.name = "nvmlDeviceGetHandleBySerial"},
    {.name = "nvmlDeviceGetHandleByUUID"},
    {.name = "nvmlDeviceGetIndex"},
    {.name = "nvmlDeviceGetInforomConfigurationChecksum"},
    {.name = "nvmlDeviceGetInforomImageVersion"},
    {.name = "nvmlDeviceGetInforomVersion"},
    {.name = "nvmlDeviceGetMaxClockInfo"},
    {.name = "nvmlDeviceGetMaxCustomerBoostClock"},
    {.name = "nvmlDeviceGetMaxPcieLinkGeneration"},
    {.name = "nvmlDeviceGetMaxPcieLinkWidth"},
    {.name = "nvmlDeviceGetMemoryErrorCounter"},
    {.name = "nvmlDeviceGetMemoryInfo"},
    {.name = "nvmlDeviceGetMinorNumber"},
    {.name = "nvmlDeviceGetMPSComputeRunningProcesses"},
    {.name = "nvmlDeviceGetMultiGpuBoard"},
    {.name = "nvmlDeviceGetName"},
    {.name = "nvmlDeviceGetNvLinkCapability"},
    {.name = "nvmlDeviceGetNvLinkErrorCounter"},
    {.name = "nvmlDeviceGetNvLinkRemotePciInfo"},
    {.name = "nvmlDeviceGetNvLinkRemotePciInfo_v2"},
    {.name = "nvmlDeviceGetNvLinkState"},
    {.name = "nvmlDeviceGetNvLinkUtilizationControl"},
    {.name = "nvmlDeviceGetNvLinkUtilizationCounter"},
    {.name = "nvmlDeviceGetNvLinkVersion"},
    {.name = "nvmlDeviceGetP2PStatus"},
    {.name = "nvmlDeviceGetPcieReplayCounter"},
    {.name = "nvmlDeviceGetPcieThroughput"},
    {.name = "nvmlDeviceGetPciInfo_v2"},
    {.name = "nvmlDeviceGetPciInfo_v3"},
    {.name = "nvmlDeviceGetPerformanceState"},
    {.name = "nvmlDeviceGetPersistenceMode"},
    {.name = "nvmlDeviceGetPowerManagementDefaultLimit"},
    {.name = "nvmlDeviceGetPowerManagementLimit"},
    {.name = "nvmlDeviceGetPowerManagementLimitConstraints"},
    {.name = "nvmlDeviceGetPowerManagementMode"},
    {.name = "nvmlDeviceGetPowerState"},
    {.name = "nvmlDeviceGetPowerUsage"},
    {.name = "nvmlDeviceGetRetiredPages"},
    {.name = "nvmlDeviceGetRetiredPagesPendingStatus"},
    {.name = "nvmlDeviceGetSamples"},
    {.name = "nvmlDeviceGetSerial"},
    {.name = "nvmlDeviceGetSupportedClocksThrottleReasons"},
    {.name = "nvmlDeviceGetSupportedEventTypes"},
    {.name = "nvmlDeviceGetSupportedGraphicsClocks"},
    {.name = "nvmlDeviceGetSupportedMemoryClocks"},
    {.name = "nvmlDeviceGetSupportedVgpus"},
    {.name = "nvmlDeviceGetTemperature"},
    {.name = "nvmlDeviceGetTemperatureThreshold"},
    {.name = "nvmlDeviceGetTopologyCommonAncestor"},
    {.name = "nvmlDeviceGetTopologyNearestGpus"},
    {.name = "nvmlDeviceGetTotalEccErrors"},
    {.name = "nvmlDeviceGetTotalEnergyConsumption"},
    {.name = "nvmlDeviceGetUtilizationRates"},
    {.name = "nvmlDeviceGetUUID"},
    {.name = "nvmlDeviceGetVbiosVersion"},
    {.name = "nvmlDeviceGetVgpuMetadata"},
    {.name = "nvmlDeviceGetVgpuProcessUtilization"},
    {.name = "nvmlDeviceGetVgpuUtilization"},
    {.name = "nvmlDeviceGetViolationStatus"},
    {.name = "nvmlDeviceGetVirtualizationMode"},
    {.name = "nvmlDeviceModifyDrainState"},
    {.name = "nvmlDeviceOnSameBoard"},
    {.name = "nvmlDeviceQueryDrainState"},
    {.name = "nvmlDeviceRegisterEvents"},
    {.name = "nvmlDeviceRemoveGpu"},
    {.name = "nvmlDeviceRemoveGpu_v2"},
    {.name = "nvmlDeviceResetApplicationsClocks"},
    {.name = "nvmlDeviceResetNvLinkErrorCounters"},
    {.name = "nvmlDeviceResetNvLinkUtilizationCounter"},
    {.name = "nvmlDeviceSetAccountingMode"},
    {.name = "nvmlDeviceSetAPIRestriction"},
    {.name = "nvmlDeviceSetApplicationsClocks"},
    {.name = "nvmlDeviceSetAutoBoostedClocksEnabled"},
    /** We hijack this call*/
    {.name = "nvmlDeviceSetComputeMode"},
    {.name = "nvmlDeviceSetCpuAffinity"},
    {.name = "nvmlDeviceSetDefaultAutoBoostedClocksEnabled"},
    {.name = "nvmlDeviceSetDriverModel"},
    {.name = "nvmlDeviceSetEccMode"},
    {.name = "nvmlDeviceSetGpuOperationMode"},
    {.name = "nvmlDeviceSetNvLinkUtilizationControl"},
    {.name = "nvmlDeviceSetPersistenceMode"},
    {.name = "nvmlDeviceSetPowerManagementLimit"},
    {.name = "nvmlDeviceSetVirtualizationMode"},
    {.name = "nvmlDeviceValidateInforom"},
    {.name = "nvmlEventSetCreate"},
    {.name = "nvmlEventSetFree"},
    {.name = "nvmlEventSetWait"},
    {.name = "nvmlGetVgpuCompatibility"},
    {.name = "nvmlInit_v2"},
    {.name = "nvmlInitWithFlags"},
    {.name = "nvmlInternalGetExportTable"},
    {.name = "nvmlSystemGetCudaDriverVersion"},
    {.name = "nvmlSystemGetCudaDriverVersion_v2"},
    {.name = "nvmlSystemGetDriverVersion"},
    {.name = "nvmlSystemGetHicVersion"},
    {.name = "nvmlSystemGetNVMLVersion"},
    {.name = "nvmlSystemGetProcessName"},
    {.name = "nvmlSystemGetTopologyGpuSet"},
    {.name = "nvmlUnitGetCount"},
    {.name = "nvmlUnitGetDevices"},
    {.name = "nvmlUnitGetFanSpeedInfo"},
    {.name = "nvmlUnitGetHandleByIndex"},
    {.name = "nvmlUnitGetLedState"},
    {.name = "nvmlUnitGetPsuInfo"},
    {.name = "nvmlUnitGetTemperature"},
    {.name = "nvmlUnitGetUnitInfo"},
    {.name = "nvmlUnitSetLedState"},
    {.name = "nvmlVgpuInstanceGetEncoderCapacity"},
    {.name = "nvmlVgpuInstanceGetEncoderSessions"},
    {.name = "nvmlVgpuInstanceGetEncoderStats"},
    {.name = "nvmlVgpuInstanceGetFbUsage"},
    {.name = "nvmlVgpuInstanceGetFrameRateLimit"},
    {.name = "nvmlVgpuInstanceGetLicenseStatus"},
    {.name = "nvmlVgpuInstanceGetMetadata"},
    {.name = "nvmlVgpuInstanceGetType"},
    {.name = "nvmlVgpuInstanceGetUUID"},
    {.name = "nvmlVgpuInstanceGetVmDriverVersion"},
    {.name = "nvmlVgpuInstanceGetVmID"},
    {.name = "nvmlVgpuInstanceSetEncoderCapacity"},
    {.name = "nvmlVgpuTypeGetClass"},
    {.name = "nvmlVgpuTypeGetDeviceID"},
    {.name = "nvmlVgpuTypeGetFramebufferSize"},
    {.name = "nvmlVgpuTypeGetFrameRateLimit"},
    {.name = "nvmlVgpuTypeGetLicense"},
    {.name = "nvmlVgpuTypeGetMaxInstances"},
    {.name = "nvmlVgpuTypeGetName"},
    {.name = "nvmlVgpuTypeGetNumDisplayHeads"},
    {.name = "nvmlVgpuTypeGetResolution"},
    {.name = "nvmlDeviceGetFBCSessions"},
    {.name = "nvmlDeviceGetFBCStats"},
    {.name = "nvmlDeviceGetGridLicensableFeatures_v2"},
    {.name = "nvmlDeviceGetRetiredPages_v2"},
    {.name = "nvmlDeviceResetGpuLockedClocks"},
    {.name = "nvmlDeviceSetGpuLockedClocks"},
    {.name = "nvmlGetBlacklistDeviceCount"},
    {.name = "nvmlGetBlacklistDeviceInfoByIndex"},
    {.name = "nvmlVgpuInstanceGetAccountingMode"},
    {.name = "nvmlVgpuInstanceGetAccountingPids"},
    {.name = "nvmlVgpuInstanceGetAccountingStats"},
    {.name = "nvmlVgpuInstanceGetFBCSessions"},
    {.name = "nvmlVgpuInstanceGetFBCStats"},
    {.name = "nvmlVgpuTypeGetMaxInstancesPerVm"},
    {.name = "nvmlGetVgpuVersion"},
    {.name = "nvmlSetVgpuVersion"},
    {.name = "nvmlDeviceGetGridLicensableFeatures_v3"},
    {.name = "nvmlDeviceGetHostVgpuMode"},
    {.name = "nvmlDeviceGetPgpuMetadataString"},
    {.name = "nvmlVgpuInstanceGetEccMode"},
    {.name = "nvmlComputeInstanceDestroy"},
    {.name = "nvmlComputeInstanceGetInfo"},
    {.name = "nvmlDeviceCreateGpuInstance"},
    {.name = "nvmlDeviceGetArchitecture"},
    {.name = "nvmlDeviceGetAttributes"},
    {.name = "nvmlDeviceGetAttributes_v2"},
    {.name = "nvmlDeviceGetComputeInstanceId"},
    {.name = "nvmlDeviceGetCpuAffinityWithinScope"},
    {.name = "nvmlDeviceGetDeviceHandleFromMigDeviceHandle"},
    {.name = "nvmlDeviceGetGpuInstanceById"},
    {.name = "nvmlDeviceGetGpuInstanceId"},
    {.name = "nvmlDeviceGetGpuInstancePossiblePlacements"},
    {.name = "nvmlDeviceGetGpuInstanceProfileInfo"},
    {.name = "nvmlDeviceGetGpuInstanceRemainingCapacity"},
    {.name = "nvmlDeviceGetGpuInstances"},
    {.name = "nvmlDeviceGetMaxMigDeviceCount"},
    {.name = "nvmlDeviceGetMemoryAffinity"},
    {.name = "nvmlDeviceGetMigDeviceHandleByIndex"},
    {.name = "nvmlDeviceGetMigMode"},
    {.name = "nvmlDeviceGetRemappedRows"},
    {.name = "nvmlDeviceGetRowRemapperHistogram"},
    {.name = "nvmlDeviceIsMigDeviceHandle"},
    {.name = "nvmlDeviceSetMigMode"},
    {.name = "nvmlEventSetWait_v2"},
    {.name = "nvmlGpuInstanceCreateComputeInstance"},
    {.name = "nvmlGpuInstanceDestroy"},
    {.name = "nvmlGpuInstanceGetComputeInstanceById"},
    {.name = "nvmlGpuInstanceGetComputeInstanceProfileInfo"},
    {.name = "nvmlGpuInstanceGetComputeInstanceRemainingCapacity"},
    {.name = "nvmlGpuInstanceGetComputeInstances"},
    {.name = "nvmlGpuInstanceGetInfo"},
    {.name = "nvmlVgpuInstanceClearAccountingPids"},
    {.name = "nvmlVgpuInstanceGetMdevUUID"},
};

static void UNUSED bug_on() {
  BUILD_BUG_ON((sizeof(nvml_library_entry) / sizeof(nvml_library_entry[0])) !=
               NVML_ENTRY_END);

  BUILD_BUG_ON((sizeof(cuda_library_entry) / sizeof(cuda_library_entry[0])) !=
               CUDA_ENTRY_END);
}

/** register once set */
static pthread_once_t g_cuda_set = PTHREAD_ONCE_INIT;
// static char base_dir[FILENAME_MAX] = EMPTY_PREFIX;
// char config_path[FILENAME_MAX] = CONTROLLER_CONFIG_PATH;
// char pid_path[FILENAME_MAX] = PIDS_CONFIG_PATH;

// char driver_version[FILENAME_MAX] = "450.51.06";
// char driver_version[FILENAME_MAX] = "535.154.05";
char driver_version[FILENAME_MAX] = "515.105.01";

static void load_driver_libraries() {
  void *table = NULL;
  char driver_filename[FILENAME_MAX];
  int i;

  snprintf(driver_filename, FILENAME_MAX - 1, "%s.%s", DRIVER_ML_LIBRARY_PREFIX,
           driver_version);
  driver_filename[FILENAME_MAX - 1] = '\0';

  table = dlopen(driver_filename, RTLD_NOW | RTLD_NODELETE);
  if (unlikely(!table)) {
    LOGGER(FATAL, "can't find library %s", driver_filename);
  }

  for (i = 0; i < NVML_ENTRY_END; i++) {
    nvml_library_entry[i].fn_ptr = dlsym(table, nvml_library_entry[i].name);
    if (unlikely(!nvml_library_entry[i].fn_ptr)) {
      LOGGER(4, "can't find function %s in %s", nvml_library_entry[i].name,
             driver_filename);
    }
  }

  dlclose(table);
}

static void load_cuda_single_library(int idx) {
  void *table = NULL;
  char cuda_filename[FILENAME_MAX];

  snprintf(cuda_filename, FILENAME_MAX - 1, "%s.%s", CUDA_LIBRARY_PREFIX,
           driver_version);
  cuda_filename[FILENAME_MAX - 1] = '\0';

  table = dlopen(cuda_filename, RTLD_NOW | RTLD_NODELETE);
  if (unlikely(!table)) {
    LOGGER(FATAL, "can't find library %s", cuda_filename);
  }

  cuda_library_entry[idx].fn_ptr = dlsym(table, cuda_library_entry[idx].name);
  if (unlikely(!cuda_library_entry[idx].fn_ptr)) {
    LOGGER(4, "can't find function %s in %s", cuda_library_entry[idx].name,
           cuda_filename);
  }

  dlclose(table);
}

void load_cuda_libraries() {
  void *table = NULL;
  int i = 0;
  char cuda_filename[FILENAME_MAX];

  LOGGER(4, "Start hijacking");

  snprintf(cuda_filename, FILENAME_MAX - 1, "%s.%s", CUDA_LIBRARY_PREFIX,
           driver_version);
  cuda_filename[FILENAME_MAX - 1] = '\0';
  cuda_filename[FILENAME_MAX - 1] = '\0';

  table = dlopen(cuda_filename, RTLD_NOW | RTLD_NODELETE);
  if (unlikely(!table)) {
    LOGGER(FATAL, "can't find library %s", cuda_filename);
  }

  for (i = 0; i < CUDA_ENTRY_END; i++) {
    cuda_library_entry[i].fn_ptr = dlsym(table, cuda_library_entry[i].name);
    if (unlikely(!cuda_library_entry[i].fn_ptr)) {
      LOGGER(4, "can't find function %s in %s", cuda_library_entry[i].name,
             cuda_filename);
    }
  }

  dlclose(table);
}

static void matchRegex(const char *pattern, const char *matchString,
                       char *version) {
  regex_t regex;
  int reti;
  regmatch_t matches[1];
  char msgbuf[512];

  reti = regcomp(&regex, pattern, REG_EXTENDED);
  if (reti) {
    LOGGER(4, "Could not compile regex: %s", DRIVER_VERSION_MATCH_PATTERN);
    return;
  }

  reti = regexec(&regex, matchString, 1, matches, 0);
  switch (reti) {
    case 0:
      strncpy(version, matchString + matches[0].rm_so,
              matches[0].rm_eo - matches[0].rm_so);
      version[matches[0].rm_eo - matches[0].rm_so] = '\0';
      break;
    case REG_NOMATCH:
      LOGGER(4, "Regex does not match for string: %s", matchString);
      break;
    default:
      regerror(reti, &regex, msgbuf, sizeof(msgbuf));
      LOGGER(4, "Regex match failed: %s", msgbuf);
  }

  regfree(&regex);
  return;
}

static void read_version_from_proc(char *version) {
  char *line = NULL;
  size_t len = 0;

  FILE *fp = fopen(DRIVER_VERSION_PROC_PATH, "r");
  if (fp == NULL) {
    LOGGER(4, "can't open %s, error %s", DRIVER_VERSION_PROC_PATH,
           strerror(errno));
    return;
  }

  while ((getline(&line, &len, fp) != -1)) {
    if (strncmp(line, "NVRM", 4) == 0) {
      matchRegex(DRIVER_VERSION_MATCH_PATTERN, line, version);
      break;
    }
  }
  fclose(fp);
}

// static int is_default_config_path() {
//   int fd = -1;

//   fd = open(config_path, O_RDONLY);
//   if (fd == -1) {
//     return 0;
//   }

//   close(fd);

//   return 1;
// }


// int read_controller_configuration() {
//   int fd = 0;
//   int rsize;
//   int ret = 1;

//   if (!is_default_config_path()) {
//     if (get_path_by_cgroup("/proc/self/cgroup")) {
//       LOGGER(FATAL, "can't get config file path");
//     }
//   }

//   fd = open(config_path, O_RDONLY);
//   if (unlikely(fd == -1)) {
//     LOGGER(4, "can't open %s, error %s", config_path, strerror(errno));
//     goto DONE;
//   }

//   rsize = (int)read(fd, (void *)&g_vcuda_config, sizeof(resource_data_t));
//   if (unlikely(rsize != sizeof(g_vcuda_config))) {
//     LOGGER(4, "can't read %s, need %zu but got %d", CONTROLLER_CONFIG_PATH,
//            sizeof(resource_data_t), rsize);
//     goto DONE;
//   }

//   read_version_from_proc(driver_version);
//   ret = 0;

//   LOGGER(4, "pod uid          : %s", g_vcuda_config.pod_uid);
//   LOGGER(4, "limit            : %d", g_vcuda_config.limit);
//   LOGGER(4, "container name   : %s", g_vcuda_config.container_name);
//   LOGGER(4, "total utilization: %d", g_vcuda_config.utilization);
//   LOGGER(4, "total gpu memory : %" PRIu64, g_vcuda_config.gpu_memory);
//   LOGGER(4, "driver version   : %s", driver_version);
//   LOGGER(4, "hard limit mode  : %d", g_vcuda_config.hard_limit);
//   LOGGER(4, "enable mode      : %d", g_vcuda_config.enable);
// DONE:
//   if (likely(fd)) {
//     close(fd);
//   }

//   return ret;
// }


void load_sim(){
  read_version_from_proc(driver_version);
  load_cuda_single_library(CUDA_ENTRY_ENUM(cuDriverGetVersion));
  load_cuda_libraries();
  load_driver_libraries();
}
void load_necessary_data() {
  pthread_once(&g_cuda_set, load_sim);
}