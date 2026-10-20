#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    VkApplicationInfo app = { .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "video2door-device-audit", .apiVersion = VK_API_VERSION_1_1 };
    VkInstanceCreateInfo info = { .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &app };
    VkInstance instance;
    VkResult result = vkCreateInstance(&info, NULL, &instance);
    if (result != VK_SUCCESS) { printf("vkCreateInstance: %d\n", result); return 1; }
    uint32_t count = 0;
    vkEnumeratePhysicalDevices(instance, &count, NULL);
    VkPhysicalDevice *devices = calloc(count, sizeof(*devices));
    vkEnumeratePhysicalDevices(instance, &count, devices);
    for (uint32_t i = 0; i < count; ++i) {
        VkPhysicalDevicePCIBusInfoPropertiesEXT pci = {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PCI_BUS_INFO_PROPERTIES_EXT };
        VkPhysicalDeviceIDProperties ids = {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES, .pNext = &pci };
        VkPhysicalDeviceProperties2 props = {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &ids };
        vkGetPhysicalDeviceProperties2(devices[i], &props);
        printf("Vulkan %u: %s vendor=%04x device=%04x pci=%04x:%02x:%02x.%x UUID=",
            i, props.properties.deviceName, props.properties.vendorID, props.properties.deviceID,
            pci.pciDomain, pci.pciBus, pci.pciDevice, pci.pciFunction);
        for (unsigned j = 0; j < VK_UUID_SIZE; ++j) printf("%02x", ids.deviceUUID[j]);
        puts("");
    }
    free(devices);
    vkDestroyInstance(instance, NULL);
    return 0;
}

