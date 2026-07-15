// Vulkan instance layer that hides VK_EXT_present_timing from applications.
//
// Why: NVIDIA's Wayland WSI (as of driver 610.x) advertises
// VK_EXT_present_timing device-wide and reports its per-surface feature,
// but reports presentTimingSupported=0 for surfaces on compositors it
// doesn't (for undocumented reasons) accept — then, when an app requests
// present timing anyway, writes timing results through an unallocated NULL
// buffer and SIGSEGVs. Under Wine/Proton that surfaces as the fatal
// `vkGetPastPresentationTimingEXT` assert. There is no compositor-side
// protocol that flips the gate (verified: matched KWin's protocol set,
// still crashes), so we simply hide the extension until the driver is
// fixed: an app that never sees VK_EXT_present_timing never enables it and
// never trips the crash.
//
// Interception is instance-level only (all three touch-points are physical-
// device queries):
//   - vkEnumerateDeviceExtensionProperties  -> drop VK_EXT_present_timing
//   - vkGetPhysicalDeviceFeatures2[KHR]      -> clear presentTiming* features
//   - vkGetPhysicalDeviceSurfaceCapabilities2KHR -> clear presentTimingSupported
#include <vulkan/vulkan.h>
#include <vulkan/vk_layer.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

// The VK_EXT_present_timing structs we scrub (VkPhysicalDevicePresentTimingFeaturesEXT,
// VkPresentTimingSurfaceCapabilitiesEXT) landed in the Vulkan headers at spec
// version 3. Fail loudly rather than silently miscompile against older headers.
#if !defined(VK_EXT_PRESENT_TIMING_SPEC_VERSION) || VK_EXT_PRESENT_TIMING_SPEC_VERSION < 3
#error "VK_EXT_present_timing v3 headers required (vulkan-headers >= 1.4.313)"
#endif

#define PT_EXT_NAME VK_EXT_PRESENT_TIMING_EXTENSION_NAME

// Minimal generic pNext walker header.
typedef struct BaseOut { VkStructureType sType; void *pNext; } BaseOut;

// Per-instance dispatch, keyed by the instance's loader dispatch pointer.
struct instance_data {
    void *key;
    PFN_vkGetInstanceProcAddr next_gipa;
    PFN_vkDestroyInstance next_destroy;
    PFN_vkEnumerateDeviceExtensionProperties next_enum_dev_ext;
    PFN_vkGetPhysicalDeviceFeatures2 next_features2;
    PFN_vkGetPhysicalDeviceFeatures2 next_features2_khr;
    PFN_vkGetPhysicalDeviceSurfaceCapabilities2KHR next_surf_caps2;
    struct instance_data *next;
};

static struct instance_data *g_instances = NULL;
static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;

// The loader-set dispatch pointer is the first word of a dispatchable handle.
static void *disp_key(void *h) { return *(void **)h; }

static struct instance_data *inst_lookup(void *handle) {
    void *k = disp_key(handle);
    pthread_mutex_lock(&g_lock);
    for (struct instance_data *d = g_instances; d; d = d->next)
        if (d->key == k) { pthread_mutex_unlock(&g_lock); return d; }
    pthread_mutex_unlock(&g_lock);
    return NULL;
}

// ---- intercepts -----------------------------------------------------------

static VKAPI_ATTR VkResult VKAPI_CALL
EnumerateDeviceExtensionProperties(VkPhysicalDevice phys, const char *layer,
                                   uint32_t *pCount, VkExtensionProperties *pProps) {
    struct instance_data *d = inst_lookup(phys);
    if (!d || !d->next_enum_dev_ext)
        return VK_ERROR_INITIALIZATION_FAILED;

    // Query the full list, filter out present_timing, then answer.
    uint32_t n = 0;
    VkResult r = d->next_enum_dev_ext(phys, layer, &n, NULL);
    if (r < 0) return r;
    VkExtensionProperties *all = n ? malloc(n * sizeof(*all)) : NULL;
    if (n && !all) return VK_ERROR_OUT_OF_HOST_MEMORY;
    r = d->next_enum_dev_ext(phys, layer, &n, all);
    if (r < 0) { free(all); return r; }

    uint32_t m = 0;
    for (uint32_t i = 0; i < n; i++)
        if (strcmp(all[i].extensionName, PT_EXT_NAME) != 0) {
            if (m != i) all[m] = all[i];
            m++;
        }

    if (!pProps) { *pCount = m; free(all); return VK_SUCCESS; }
    uint32_t copy = *pCount < m ? *pCount : m;
    for (uint32_t i = 0; i < copy; i++) pProps[i] = all[i];
    *pCount = copy;
    free(all);
    return copy < m ? VK_INCOMPLETE : VK_SUCCESS;
}

static void scrub_features(void *pNext) {
    for (BaseOut *s = pNext; s; s = s->pNext) {
        if (s->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_TIMING_FEATURES_EXT) {
            VkPhysicalDevicePresentTimingFeaturesEXT *f = (void *)s;
            f->presentTiming = VK_FALSE;
            f->presentAtAbsoluteTime = VK_FALSE;
            f->presentAtRelativeTime = VK_FALSE;
        }
    }
}

static VKAPI_ATTR void VKAPI_CALL
GetPhysicalDeviceFeatures2(VkPhysicalDevice phys, VkPhysicalDeviceFeatures2 *pFeatures) {
    struct instance_data *d = inst_lookup(phys);
    if (d && d->next_features2) d->next_features2(phys, pFeatures);
    if (pFeatures) scrub_features(pFeatures->pNext);
}

static VKAPI_ATTR void VKAPI_CALL
GetPhysicalDeviceFeatures2KHR(VkPhysicalDevice phys, VkPhysicalDeviceFeatures2 *pFeatures) {
    struct instance_data *d = inst_lookup(phys);
    if (d && d->next_features2_khr) d->next_features2_khr(phys, pFeatures);
    if (pFeatures) scrub_features(pFeatures->pNext);
}

static VKAPI_ATTR VkResult VKAPI_CALL
GetPhysicalDeviceSurfaceCapabilities2KHR(VkPhysicalDevice phys,
                                         const VkPhysicalDeviceSurfaceInfo2KHR *pInfo,
                                         VkSurfaceCapabilities2KHR *pCaps) {
    struct instance_data *d = inst_lookup(phys);
    VkResult r = VK_ERROR_INITIALIZATION_FAILED;
    if (d && d->next_surf_caps2) r = d->next_surf_caps2(phys, pInfo, pCaps);
    if (r >= 0 && pCaps) {
        for (BaseOut *s = pCaps->pNext; s; s = s->pNext) {
            if (s->sType == VK_STRUCTURE_TYPE_PRESENT_TIMING_SURFACE_CAPABILITIES_EXT) {
                VkPresentTimingSurfaceCapabilitiesEXT *c = (void *)s;
                c->presentTimingSupported = VK_FALSE;
                c->presentAtAbsoluteTimeSupported = VK_FALSE;
                c->presentAtRelativeTimeSupported = VK_FALSE;
                // presentStageQueries left as-is: meaningless once
                // presentTimingSupported is false, and the app won't read it.
            }
        }
    }
    return r;
}

// ---- plumbing -------------------------------------------------------------

static VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL GetInstanceProcAddr(VkInstance inst, const char *name);

static VKAPI_ATTR VkResult VKAPI_CALL
CreateInstance(const VkInstanceCreateInfo *pCreateInfo, const VkAllocationCallbacks *pAlloc,
               VkInstance *pInstance) {
    VkLayerInstanceCreateInfo *ci = (VkLayerInstanceCreateInfo *)pCreateInfo->pNext;
    while (ci && !(ci->sType == VK_STRUCTURE_TYPE_LOADER_INSTANCE_CREATE_INFO &&
                   ci->function == VK_LAYER_LINK_INFO))
        ci = (VkLayerInstanceCreateInfo *)ci->pNext;
    if (!ci) return VK_ERROR_INITIALIZATION_FAILED;

    PFN_vkGetInstanceProcAddr next_gipa = ci->u.pLayerInfo->pfnNextGetInstanceProcAddr;
    ci->u.pLayerInfo = ci->u.pLayerInfo->pNext; // advance the chain for the next layer

    PFN_vkCreateInstance next_create =
        (PFN_vkCreateInstance)next_gipa(NULL, "vkCreateInstance");
    if (!next_create) return VK_ERROR_INITIALIZATION_FAILED;
    VkResult r = next_create(pCreateInfo, pAlloc, pInstance);
    if (r != VK_SUCCESS) return r;

    struct instance_data *d = calloc(1, sizeof(*d));
    if (!d) return VK_SUCCESS; // instance still usable, just unfiltered
    d->key = disp_key(*pInstance);
    d->next_gipa = next_gipa;
    d->next_destroy = (PFN_vkDestroyInstance)next_gipa(*pInstance, "vkDestroyInstance");
    d->next_enum_dev_ext = (PFN_vkEnumerateDeviceExtensionProperties)
        next_gipa(*pInstance, "vkEnumerateDeviceExtensionProperties");
    d->next_features2 = (PFN_vkGetPhysicalDeviceFeatures2)
        next_gipa(*pInstance, "vkGetPhysicalDeviceFeatures2");
    d->next_features2_khr = (PFN_vkGetPhysicalDeviceFeatures2)
        next_gipa(*pInstance, "vkGetPhysicalDeviceFeatures2KHR");
    d->next_surf_caps2 = (PFN_vkGetPhysicalDeviceSurfaceCapabilities2KHR)
        next_gipa(*pInstance, "vkGetPhysicalDeviceSurfaceCapabilities2KHR");
    pthread_mutex_lock(&g_lock);
    d->next = g_instances; g_instances = d;
    pthread_mutex_unlock(&g_lock);
    return VK_SUCCESS;
}

static VKAPI_ATTR void VKAPI_CALL
DestroyInstance(VkInstance inst, const VkAllocationCallbacks *pAlloc) {
    struct instance_data *d = inst_lookup(inst);
    PFN_vkDestroyInstance destroy = d ? d->next_destroy : NULL;
    if (d) {
        pthread_mutex_lock(&g_lock);
        for (struct instance_data **pp = &g_instances; *pp; pp = &(*pp)->next)
            if (*pp == d) { *pp = d->next; break; }
        pthread_mutex_unlock(&g_lock);
        free(d);
    }
    if (destroy) destroy(inst, pAlloc);
}

#define ENTRY(n, f) if (!strcmp(name, n)) return (PFN_vkVoidFunction)(f)

static VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL
GetInstanceProcAddr(VkInstance inst, const char *name) {
    ENTRY("vkGetInstanceProcAddr", GetInstanceProcAddr);
    ENTRY("vkCreateInstance", CreateInstance);
    ENTRY("vkDestroyInstance", DestroyInstance);
    ENTRY("vkEnumerateDeviceExtensionProperties", EnumerateDeviceExtensionProperties);
    ENTRY("vkGetPhysicalDeviceFeatures2", GetPhysicalDeviceFeatures2);
    ENTRY("vkGetPhysicalDeviceFeatures2KHR", GetPhysicalDeviceFeatures2KHR);
    ENTRY("vkGetPhysicalDeviceSurfaceCapabilities2KHR", GetPhysicalDeviceSurfaceCapabilities2KHR);
    struct instance_data *d = inst ? inst_lookup(inst) : NULL;
    if (d && d->next_gipa) return d->next_gipa(inst, name);
    return NULL;
}

__attribute__((visibility("default"))) VKAPI_ATTR VkResult VKAPI_CALL
vkNegotiateLoaderLayerInterfaceVersion(VkNegotiateLayerInterface *v) {
    if (v->loaderLayerInterfaceVersion > 2) v->loaderLayerInterfaceVersion = 2;
    v->pfnGetInstanceProcAddr = GetInstanceProcAddr;
    v->pfnGetDeviceProcAddr = NULL;
    v->pfnGetPhysicalDeviceProcAddr = NULL;
    return VK_SUCCESS;
}
