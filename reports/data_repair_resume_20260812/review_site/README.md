# 新修复棒球数据检索复核页

打开 `index.html` 即可检索 315 条 evidence-verified candidates，并查看每条的最终时间、门禁、来源、staged 五文件路径和本地媒体。

口径边界：

- `NEWLY_VERIFIED_MANIFEST.csv` 唯一样本数为 315。
- `READY_TO_MATERIALIZE.csv` 为 315，`BLOCKED_MATERIALIZATION.csv` 为 0。
- 当前正式 checkout 中新增“已回写最终时间、可直接训练”的行数仍按 0 计算。
- GitHub 分支只提交网页和索引，不提交 staged 视频、音频等大媒体文件；媒体播放器在本地工作区打开时可用。

复核标记保存在浏览器 `localStorage`，不会修改正式 dataset。需要留档时，在页面右上角导出 CSV 或 JSON。
