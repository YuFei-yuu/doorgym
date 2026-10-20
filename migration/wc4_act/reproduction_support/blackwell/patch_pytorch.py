from pathlib import Path
import re


root = Path('/data/zyf/video2door/.repro/blackwell/src/pytorch')
cmake = root / 'cmake/Modules_CUDA_fix/upstream/FindCUDA/select_compute_arch.cmake'
lines = cmake.read_text().splitlines(keepends=True)
matchers = [i for i, line in enumerate(lines) if 'if(arch_name MATCHES "^(' in line]
assert len(matchers) == 1, 'Unexpected architecture matcher'
i = matchers[0]
lines[i] = re.sub(r'\[0-9\]\+*', '[0-9]+', lines[i])
cmake.write_text(''.join(lines))

extension = root / 'torch/utils/cpp_extension.py'
text = extension.read_text()
needle = "'9.0', '9.0a']"
replacement = "'9.0', '9.0a', '12.0']"
if replacement not in text:
    assert text.count(needle) == 1, 'Unexpected supported_arches list'
    text = text.replace(needle, replacement)
    extension.write_text(text)

cudnn = root / 'cmake/Modules_CUDA_fix/FindCUDNN.cmake'
text = cudnn.read_text()
needle = 'find_library(CUDNN_LIBRARY_PATH ${CUDNN_LIBNAME}'
replacement = '''# Honor an explicit library file, including versioned pip cuDNN libraries.
if(EXISTS "${CUDNN_LIBRARY}" AND NOT IS_DIRECTORY "${CUDNN_LIBRARY}")
  set(CUDNN_LIBRARY_PATH "${CUDNN_LIBRARY}" CACHE FILEPATH "cuDNN library" FORCE)
endif()

find_library(CUDNN_LIBRARY_PATH ${CUDNN_LIBNAME}'''
if replacement not in text:
    assert text.count(needle) == 1
    cudnn.write_text(text.replace(needle, replacement))
print('Added numeric compute capability 12.0 to CMake and C++ extension validation.')

