# 取回归档

普通 clone 可能受本机 Git 换行设置影响。要保持与原文件逐字节一致，按以下步骤取回（将 URL 和目录替换为目标仓库）：

```powershell
git -c core.autocrlf=false clone --no-checkout https://github.com/ff0809/REPOSITORY.git REPOSITORY
Set-Location REPOSITORY
Set-Content -LiteralPath .git/info/attributes -Value '* -text -filter -ident' -Encoding ascii
git read-tree HEAD
git -c core.autocrlf=false checkout-index --all --force
python -m pip install requests
python _archive/restore_archive.py
```

私有仓库需要当前账号有读取权限。恢复脚本仅通过 Git Credential Manager 取得当前 GitHub 凭证，不打印或保存令牌。下载使用 GitHub Release API，并校验每个分片、完整文件及清单中的所有文件。

原始 Git 历史的分片映射见 excluded.json。恢复脚本会按顺序拼接并验证 bundle；可用 `git clone PATH_TO_BUNDLE TARGET_DIR` 取回原仓库历史。

虚拟环境、node_modules、IDE 索引缓存可由依赖清单和工程重新生成。原始 Git 历史与当前文件快照分别保存。
