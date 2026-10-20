/*
 * Instance-layer chaining follows Mesa's MIT-licensed device_select_layer.c:
 * Copyright (c) 2017 Google; Copyright (c) 2019 Red Hat.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * This process-local layer exposes one audited physical GPU as Vulkan device 0.
 * It changes device enumeration only, leaving rendering and memory APIs intact.
 */
#include <vulkan/vk_layer.h>
#include <vulkan/vulkan.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>

typedef struct Context {
    VkInstance instance;
    PFN_vkGetInstanceProcAddr gipa;
    PFN_vkDestroyInstance destroy;
    PFN_vkEnumeratePhysicalDevices enumerate;
    struct Context *next;
} Context;
static Context *contexts;
static pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

static Context *context_for(VkInstance instance) {
    pthread_mutex_lock(&mutex);
    Context *ctx = contexts;
    while (ctx && ctx->instance != instance) ctx = ctx->next;
    pthread_mutex_unlock(&mutex);
    return ctx;
}

static VkResult selected_device(Context *ctx, VkPhysicalDevice *selected) {
    const char *setting = getenv("VIDEO2DOOR_VK_DEVICE_INDEX");
    if (!setting || !*setting) return VK_ERROR_INITIALIZATION_FAILED;
    char *end;
    long index = strtol(setting, &end, 10);
    if (*end || index < 0) return VK_ERROR_INITIALIZATION_FAILED;
    uint32_t count = 0;
    VkResult result = ctx->enumerate(ctx->instance, &count, NULL);
    if (result != VK_SUCCESS) return result;
    if ((unsigned long)index >= count) return VK_ERROR_INITIALIZATION_FAILED;
    VkPhysicalDevice *all = calloc(count, sizeof(*all));
    if (!all) return VK_ERROR_OUT_OF_HOST_MEMORY;
    result = ctx->enumerate(ctx->instance, &count, all);
    if (result == VK_SUCCESS && (unsigned long)index < count) *selected = all[index];
    else if (result == VK_SUCCESS) result = VK_ERROR_INITIALIZATION_FAILED;
    free(all);
    return result;
}

static VKAPI_ATTR VkResult VKAPI_CALL filter_enumerate(VkInstance instance,
        uint32_t *count, VkPhysicalDevice *devices) {
    Context *ctx = context_for(instance);
    if (!ctx) return VK_ERROR_INITIALIZATION_FAILED;
    VkPhysicalDevice selected;
    VkResult result = selected_device(ctx, &selected);
    if (result != VK_SUCCESS) return result;
    if (!devices) { *count = 1; return VK_SUCCESS; }
    if (*count == 0) return VK_INCOMPLETE;
    devices[0] = selected;
    *count = 1;
    return VK_SUCCESS;
}

static VKAPI_ATTR VkResult VKAPI_CALL filter_groups(VkInstance instance,
        uint32_t *count, VkPhysicalDeviceGroupProperties *groups) {
    Context *ctx = context_for(instance);
    if (!ctx) return VK_ERROR_INITIALIZATION_FAILED;
    VkPhysicalDevice selected;
    VkResult result = selected_device(ctx, &selected);
    if (result != VK_SUCCESS) return result;
    if (!groups) { *count = 1; return VK_SUCCESS; }
    if (*count == 0) return VK_INCOMPLETE;
    groups[0].physicalDeviceCount = 1;
    groups[0].physicalDevices[0] = selected;
    groups[0].subsetAllocation = VK_FALSE;
    *count = 1;
    return VK_SUCCESS;
}

static VKAPI_ATTR VkResult VKAPI_CALL filter_create(const VkInstanceCreateInfo *info,
        const VkAllocationCallbacks *allocator, VkInstance *instance) {
    VkLayerInstanceCreateInfo *chain = (void *)info->pNext;
    while (chain && !(chain->sType == VK_STRUCTURE_TYPE_LOADER_INSTANCE_CREATE_INFO &&
                     chain->function == VK_LAYER_LINK_INFO)) chain = (void *)chain->pNext;
    if (!chain || !chain->u.pLayerInfo) return VK_ERROR_INITIALIZATION_FAILED;
    PFN_vkGetInstanceProcAddr next_gipa = chain->u.pLayerInfo->pfnNextGetInstanceProcAddr;
    PFN_vkCreateInstance next_create = (PFN_vkCreateInstance)next_gipa(NULL, "vkCreateInstance");
    if (!next_create) return VK_ERROR_INITIALIZATION_FAILED;
    Context *ctx = calloc(1, sizeof(*ctx));
    if (!ctx) return VK_ERROR_OUT_OF_HOST_MEMORY;
    chain->u.pLayerInfo = chain->u.pLayerInfo->pNext;
    VkResult result = next_create(info, allocator, instance);
    if (result != VK_SUCCESS) { free(ctx); return result; }
    ctx->instance = *instance;
    ctx->gipa = next_gipa;
    ctx->destroy = (PFN_vkDestroyInstance)next_gipa(*instance, "vkDestroyInstance");
    ctx->enumerate = (PFN_vkEnumeratePhysicalDevices)next_gipa(*instance, "vkEnumeratePhysicalDevices");
    pthread_mutex_lock(&mutex);
    ctx->next = contexts;
    contexts = ctx;
    pthread_mutex_unlock(&mutex);
    return VK_SUCCESS;
}

static VKAPI_ATTR void VKAPI_CALL filter_destroy(VkInstance instance,
        const VkAllocationCallbacks *allocator) {
    pthread_mutex_lock(&mutex);
    Context **link = &contexts;
    while (*link && (*link)->instance != instance) link = &(*link)->next;
    Context *ctx = *link;
    if (ctx) *link = ctx->next;
    pthread_mutex_unlock(&mutex);
    if (ctx) { ctx->destroy(instance, allocator); free(ctx); }
}

static VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL filter_gipa(VkInstance instance,
        const char *name) {
    if (!strcmp(name, "vkGetInstanceProcAddr")) return (PFN_vkVoidFunction)filter_gipa;
    if (!strcmp(name, "vkCreateInstance")) return (PFN_vkVoidFunction)filter_create;
    if (!strcmp(name, "vkDestroyInstance")) return (PFN_vkVoidFunction)filter_destroy;
    if (!strcmp(name, "vkEnumeratePhysicalDevices")) return (PFN_vkVoidFunction)filter_enumerate;
    if (!strcmp(name, "vkEnumeratePhysicalDeviceGroups") ||
        !strcmp(name, "vkEnumeratePhysicalDeviceGroupsKHR")) return (PFN_vkVoidFunction)filter_groups;
    Context *ctx = context_for(instance);
    return ctx ? ctx->gipa(instance, name) : NULL;
}

__attribute__((visibility("default"))) VKAPI_ATTR VkResult VKAPI_CALL
vkNegotiateLoaderLayerInterfaceVersion(VkNegotiateLayerInterface *version) {
    if (version->loaderLayerInterfaceVersion < 2) return VK_ERROR_INITIALIZATION_FAILED;
    version->loaderLayerInterfaceVersion = 2;
    version->pfnGetInstanceProcAddr = filter_gipa;
    return VK_SUCCESS;
}

