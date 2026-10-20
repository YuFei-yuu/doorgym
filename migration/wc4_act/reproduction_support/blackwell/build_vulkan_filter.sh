#!/usr/bin/env bash
set -euo pipefail
root=/data/zyf/video2door/.repro/blackwell
mkdir -p "$root/vulkan-layer"
/usr/bin/gcc -Wall -Wextra -O2 "$root/probe_vulkan.c" -lvulkan -o "$root/probe_vulkan"
/usr/bin/gcc -shared -fPIC -fvisibility=hidden -Wall -Wextra -O2 \
    "$root/vulkan_device_filter.c" -lpthread -o "$root/libvulkan_video2door_device_select.so"
cp "$root/VkLayer_video2door_device_select.json" "$root/vulkan-layer/"

