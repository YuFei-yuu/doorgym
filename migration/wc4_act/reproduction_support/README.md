# 复现手册环境附件

与相邻的 [REPRODUCTION_GUIDE.md](../REPRODUCTION_GUIDE.md) 一同交付。本目录提供从零重建环境需要的小型支持文件，不包含模型、视频、环境二进制或凭据。

- `blackwell/`：从 pro5000 `/data/zyf/video2door/.repro/blackwell/` 原样复制的实际构建、补丁、Vulkan设备筛选源码与历史依赖清单。
- `act/act-conda-explicit.txt`、`act/act-pip-freeze.txt`：已验收50条专家数据采集阶段的环境导出，包含修复后的FFmpeg环境。
- `act/ffmpeg-additions-explicit.txt`：当时修复所新增42包的精确URL清单，仅供差异/恢复追溯。
- `act/requirements-pinned.txt`：从ACT freeze派生，去掉editable Git URL；定制LeRobot由当前本地源码单独安装。
- `blackwell/requirements-build-and-runtime.txt`：从Gym freeze派生，去掉local editable及单独安装的Torch/torchvision/torchaudio；其余版本保留。
- `manifest.json`：上述19份输入文件的来源、大小和SHA256。该清单不包含本说明文件及清单自身。

原样支持脚本保留 `/data/zyf/video2door` 等实际路径；使用它们前按主手册建立该支持布局。在其他目录部署时，只适配环境支持路径，并重新核对CUDA/Vulkan设备绑定。

本次只复制并检查文件，未安装包、重编译Torch或启动仿真。主手册列明了新机器的环境准入检查；不要将静态文档校验称为已经完成空白主机重建。

在本目录执行以下命令检查附件：

```bash
python - <<'PY'
import hashlib, json
from pathlib import Path
root = Path(".")
manifest = json.loads((root/"manifest.json").read_text())
for name, receipt in manifest["files"].items():
    payload = (root/name).read_bytes()
    assert len(payload) == receipt["bytes"], name
    assert hashlib.sha256(payload).hexdigest() == receipt["sha256"], name
print("PASS:", len(manifest["files"]), "reproduction support files")
PY
```
