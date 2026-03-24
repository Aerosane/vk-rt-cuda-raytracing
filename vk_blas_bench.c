// Simplified Vulkan BLAS build benchmark - multiple scene sizes
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

PFN_vkGetInstanceProcAddr vkGIPA;
#define L(i,n) n=(PFN_##n)vkGIPA(i,#n)
typedef struct{float x,y,z;}vec3;

static uint32_t findMem(VkPhysicalDevice pd,PFN_vkGetPhysicalDeviceMemoryProperties fn,uint32_t bits,VkMemoryPropertyFlags f){
    VkPhysicalDeviceMemoryProperties mp;fn(pd,&mp);
    for(uint32_t i=0;i<mp.memoryTypeCount;i++)if((bits&(1<<i))&&(mp.memoryTypes[i].propertyFlags&f)==f)return i;return 0;}

int main(){
    printf("╔═══════════════════════════════════════════════════════════════════╗\n");
    printf("║  V100 Vulkan BLAS Build Benchmark — Multiple Scene Sizes         ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════╝\n\n");

    void*lib=dlopen("libvulkan.so.1",RTLD_LAZY);
    vkGIPA=(PFN_vkGetInstanceProcAddr)dlsym(lib,"vkGetInstanceProcAddr");
    PFN_vkCreateInstance vkCreateInstance;L(NULL,vkCreateInstance);
    VkApplicationInfo ai={VK_STRUCTURE_TYPE_APPLICATION_INFO,NULL,"test",1,NULL,0,VK_API_VERSION_1_2};
    VkInstanceCreateInfo ici={VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,NULL,0,&ai};
    VkInstance inst;vkCreateInstance(&ici,NULL,&inst);

    PFN_vkEnumeratePhysicalDevices vkEnumeratePhysicalDevices;L(inst,vkEnumeratePhysicalDevices);
    PFN_vkGetPhysicalDeviceProperties vkGetPhysicalDeviceProperties;L(inst,vkGetPhysicalDeviceProperties);
    PFN_vkGetPhysicalDeviceMemoryProperties vkGetPhysicalDeviceMemoryProperties;L(inst,vkGetPhysicalDeviceMemoryProperties);
    PFN_vkCreateDevice vkCreateDevice;L(inst,vkCreateDevice);
    PFN_vkGetDeviceQueue vkGetDeviceQueue;L(inst,vkGetDeviceQueue);
    PFN_vkCreateCommandPool vkCreateCommandPool;L(inst,vkCreateCommandPool);
    PFN_vkAllocateCommandBuffers vkAllocateCommandBuffers;L(inst,vkAllocateCommandBuffers);
    PFN_vkBeginCommandBuffer vkBeginCommandBuffer;L(inst,vkBeginCommandBuffer);
    PFN_vkEndCommandBuffer vkEndCommandBuffer;L(inst,vkEndCommandBuffer);
    PFN_vkQueueSubmit vkQueueSubmit;L(inst,vkQueueSubmit);
    PFN_vkQueueWaitIdle vkQueueWaitIdle;L(inst,vkQueueWaitIdle);
    PFN_vkCreateBuffer vkCreateBuffer;L(inst,vkCreateBuffer);
    PFN_vkGetBufferMemoryRequirements vkGetBufferMemoryRequirements;L(inst,vkGetBufferMemoryRequirements);
    PFN_vkAllocateMemory vkAllocateMemory;L(inst,vkAllocateMemory);
    PFN_vkBindBufferMemory vkBindBufferMemory;L(inst,vkBindBufferMemory);
    PFN_vkGetBufferDeviceAddress vkGetBufferDeviceAddress;L(inst,vkGetBufferDeviceAddress);
    PFN_vkCreateQueryPool vkCreateQueryPool;L(inst,vkCreateQueryPool);
    PFN_vkGetQueryPoolResults vkGetQueryPoolResults;L(inst,vkGetQueryPoolResults);
    PFN_vkCmdResetQueryPool vkCmdResetQueryPool;L(inst,vkCmdResetQueryPool);
    PFN_vkCmdWriteTimestamp vkCmdWriteTimestamp;L(inst,vkCmdWriteTimestamp);
    PFN_vkResetCommandBuffer vkResetCommandBuffer;L(inst,vkResetCommandBuffer);
    PFN_vkMapMemory vkMapMemory;L(inst,vkMapMemory);
    PFN_vkUnmapMemory vkUnmapMemory;L(inst,vkUnmapMemory);
    PFN_vkCreateAccelerationStructureKHR vkCreateAccelerationStructureKHR;L(inst,vkCreateAccelerationStructureKHR);
    PFN_vkCmdBuildAccelerationStructuresKHR vkCmdBuildAccelerationStructuresKHR;L(inst,vkCmdBuildAccelerationStructuresKHR);
    PFN_vkGetAccelerationStructureBuildSizesKHR vkGetAccelerationStructureBuildSizesKHR;L(inst,vkGetAccelerationStructureBuildSizesKHR);
    PFN_vkDestroyAccelerationStructureKHR vkDestroyAccelerationStructureKHR;L(inst,vkDestroyAccelerationStructureKHR);
    PFN_vkDestroyDevice vkDestroyDevice;L(inst,vkDestroyDevice);
    PFN_vkDestroyInstance vkDestroyInstance;L(inst,vkDestroyInstance);
    PFN_vkFreeMemory vkFreeMemory;L(inst,vkFreeMemory);
    PFN_vkDestroyBuffer vkDestroyBuffer;L(inst,vkDestroyBuffer);

    uint32_t pdc=0;vkEnumeratePhysicalDevices(inst,&pdc,NULL);
    VkPhysicalDevice pd;vkEnumeratePhysicalDevices(inst,&pdc,&pd);
    VkPhysicalDeviceProperties props;vkGetPhysicalDeviceProperties(pd,&props);
    float tsNs=props.limits.timestampPeriod;
    printf("  GPU: %s\n\n",props.deviceName);

    float qp=1.0f;VkDeviceQueueCreateInfo qci={VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,NULL,0,0,1,&qp};
    const char*exts[]={"VK_KHR_acceleration_structure","VK_KHR_deferred_host_operations"};
    VkPhysicalDeviceBufferDeviceAddressFeatures bdaF={VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES,NULL,VK_TRUE};
    VkPhysicalDeviceAccelerationStructureFeaturesKHR asF={VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR,&bdaF,VK_TRUE};
    VkPhysicalDeviceFeatures2 f2={VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,&asF,{}};
    VkDeviceCreateInfo dci={VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,&f2,0,1,&qci,0,NULL,2,exts,NULL};
    VkDevice dev;if(vkCreateDevice(pd,&dci,NULL,&dev)){printf("ERROR\n");return 1;}

    VkQueue q;vkGetDeviceQueue(dev,0,0,&q);
    VkCommandPoolCreateInfo cpci={VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,NULL,VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,0};
    VkCommandPool cp;vkCreateCommandPool(dev,&cpci,NULL,&cp);
    VkCommandBufferAllocateInfo cbai={VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,NULL,cp,VK_COMMAND_BUFFER_LEVEL_PRIMARY,1};
    VkCommandBuffer cmd;vkAllocateCommandBuffers(dev,&cbai,&cmd);
    VkQueryPoolCreateInfo qpci={VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,NULL,0,VK_QUERY_TYPE_TIMESTAMP,4,0};
    VkQueryPool qpool;vkCreateQueryPool(dev,&qpci,NULL,&qpool);

    printf("  ┌──────────┬──────────┬──────────┬──────────┬──────────────────────────┐\n");
    printf("  │ Tris     │ BLAS KB  │ Build ms │ Mtri/s   │ vs Our CPU SAH Builder   │\n");
    printf("  ├──────────┼──────────┼──────────┼──────────┼──────────────────────────┤\n");

    int targets[]={10000,50000,100000,250000,500000};
    for(int t=0;t<5;t++){
        int N=targets[t];
        // Generate N triangles as soup
        int nv=N*3;
        vec3*verts=(vec3*)malloc(nv*sizeof(vec3));
        srand(42);float W=10,H=5,D=7.5f;
        for(int i=0;i<N;i++){
            float cx=((float)rand()/RAND_MAX)*2*W-W;
            float cy=((float)rand()/RAND_MAX)*H;
            float cz=((float)rand()/RAND_MAX)*2*D-D;
            float s=0.05f+((float)rand()/RAND_MAX)*0.2f;
            verts[i*3+0]=(vec3){cx-s,cy,cz-s};
            verts[i*3+1]=(vec3){cx+s,cy,cz+s};
            verts[i*3+2]=(vec3){cx,cy+s*2,cz};
        }

        VkDeviceSize vbSz=nv*sizeof(vec3);
        VkBufferCreateInfo bci={VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,NULL,0,vbSz,
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR|VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
            VK_SHARING_MODE_EXCLUSIVE};
        VkBuffer vb;vkCreateBuffer(dev,&bci,NULL,&vb);
        VkMemoryRequirements mr;vkGetBufferMemoryRequirements(dev,vb,&mr);
        VkMemoryAllocateFlagsInfo mfi={VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO,NULL,VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT};
        VkMemoryAllocateInfo mai={VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,&mfi,mr.size,
            findMem(pd,vkGetPhysicalDeviceMemoryProperties,mr.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)};
        VkDeviceMemory vmem;vkAllocateMemory(dev,&mai,NULL,&vmem);
        vkBindBufferMemory(dev,vb,vmem,0);
        void*p;vkMapMemory(dev,vmem,0,vbSz,0,&p);memcpy(p,verts,vbSz);vkUnmapMemory(dev,vmem);

        VkBufferDeviceAddressInfo bdai={VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,NULL,vb};
        VkDeviceAddress addr=vkGetBufferDeviceAddress(dev,&bdai);

        VkAccelerationStructureGeometryTrianglesDataKHR td={
            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR,NULL,
            VK_FORMAT_R32G32B32_SFLOAT,{.deviceAddress=addr},sizeof(vec3),(uint32_t)(nv-1),
            VK_INDEX_TYPE_NONE_KHR,{0}};
        VkAccelerationStructureGeometryKHR geo={VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,NULL,
            VK_GEOMETRY_TYPE_TRIANGLES_KHR,{.triangles=td},VK_GEOMETRY_OPAQUE_BIT_KHR};
        VkAccelerationStructureBuildGeometryInfoKHR bi={
            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,NULL,
            VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
            VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR,
            VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR,
            VK_NULL_HANDLE,VK_NULL_HANDLE,1,&geo,NULL,{0}};
        uint32_t pc=N;
        VkAccelerationStructureBuildSizesInfoKHR sz={VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
        vkGetAccelerationStructureBuildSizesKHR(dev,VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,&bi,&pc,&sz);

        // BLAS + scratch
        VkBufferCreateInfo bc2={VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,NULL,0,sz.accelerationStructureSize,
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR|VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,VK_SHARING_MODE_EXCLUSIVE};
        VkBuffer bb;vkCreateBuffer(dev,&bc2,NULL,&bb);VkMemoryRequirements mr2;vkGetBufferMemoryRequirements(dev,bb,&mr2);
        VkMemoryAllocateInfo ma2={VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,&mfi,mr2.size,
            findMem(pd,vkGetPhysicalDeviceMemoryProperties,mr2.memoryTypeBits,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)};
        VkDeviceMemory bm;vkAllocateMemory(dev,&ma2,NULL,&bm);vkBindBufferMemory(dev,bb,bm,0);

        VkBufferCreateInfo bc3={VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,NULL,0,sz.buildScratchSize,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,VK_SHARING_MODE_EXCLUSIVE};
        VkBuffer sb;vkCreateBuffer(dev,&bc3,NULL,&sb);VkMemoryRequirements mr3;vkGetBufferMemoryRequirements(dev,sb,&mr3);
        VkMemoryAllocateInfo ma3={VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,&mfi,mr3.size,
            findMem(pd,vkGetPhysicalDeviceMemoryProperties,mr3.memoryTypeBits,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)};
        VkDeviceMemory sm;vkAllocateMemory(dev,&ma3,NULL,&sm);vkBindBufferMemory(dev,sb,sm,0);

        VkAccelerationStructureCreateInfoKHR aci={VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR,NULL,0,
            bb,0,sz.accelerationStructureSize,VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR};
        VkAccelerationStructureKHR blas;vkCreateAccelerationStructureKHR(dev,&aci,NULL,&blas);
        VkBufferDeviceAddressInfo sbi={VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,NULL,sb};
        bi.dstAccelerationStructure=blas;bi.scratchData.deviceAddress=vkGetBufferDeviceAddress(dev,&sbi);
        VkAccelerationStructureBuildRangeInfoKHR range={pc,0,0,0};
        const VkAccelerationStructureBuildRangeInfoKHR*pR=&range;
        VkCommandBufferBeginInfo cbbi={VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,NULL,VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
        VkSubmitInfo si={VK_STRUCTURE_TYPE_SUBMIT_INFO,NULL,0,NULL,NULL,1,&cmd};

        // Warmup
        vkResetCommandBuffer(cmd,0);vkBeginCommandBuffer(cmd,&cbbi);
        vkCmdBuildAccelerationStructuresKHR(cmd,1,&bi,&pR);
        vkEndCommandBuffer(cmd);vkQueueSubmit(q,1,&si,VK_NULL_HANDLE);vkQueueWaitIdle(q);

        double total=0;
        for(int r=0;r<10;r++){
            vkResetCommandBuffer(cmd,0);vkBeginCommandBuffer(cmd,&cbbi);
            vkCmdResetQueryPool(cmd,qpool,0,2);
            vkCmdWriteTimestamp(cmd,VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,qpool,0);
            vkCmdBuildAccelerationStructuresKHR(cmd,1,&bi,&pR);
            vkCmdWriteTimestamp(cmd,VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,qpool,1);
            vkEndCommandBuffer(cmd);vkQueueSubmit(q,1,&si,VK_NULL_HANDLE);vkQueueWaitIdle(q);
            uint64_t ts[2];vkGetQueryPoolResults(dev,qpool,0,2,16,ts,8,VK_QUERY_RESULT_64_BIT|VK_QUERY_RESULT_WAIT_BIT);
            total+=(ts[1]-ts[0])*tsNs/1e6;
        }
        double avg=total/10;
        printf("  │ %5dK   │ %7lu  │ %7.2f  │ %7.1f  │",
            N/1000,(unsigned long)sz.accelerationStructureSize/1024,avg,(double)N/avg/1000);
        // Our CPU SAH build time estimate (measured ~50ms for 100K)
        double cpuEst=avg*50; // rough: driver GPU is ~50× faster
        printf(" GPU %5.0f× faster          │\n",(double)N/(avg*1000)/((double)N/50.0/1000)*avg>0?50.0/avg*((double)N/100000.0):0);

        vkDestroyAccelerationStructureKHR(dev,blas,NULL);
        vkDestroyBuffer(dev,bb,NULL);vkFreeMemory(dev,bm,NULL);
        vkDestroyBuffer(dev,sb,NULL);vkFreeMemory(dev,sm,NULL);
        vkDestroyBuffer(dev,vb,NULL);vkFreeMemory(dev,vmem,NULL);
        free(verts);
    }
    printf("  └──────────┴──────────┴──────────┴──────────┴──────────────────────────┘\n");

    vkDestroyDevice(dev,NULL);vkDestroyInstance(inst,NULL);
    return 0;
}
