# GitHub 个人分支使用指南（VS Code）

这份指南面向第一次使用 Git 和 GitHub 的组员。请先阅读完整文档，再开始上传或修改数据。

当前项目从现在开始采用个人分支工作方式：

- `main` 分支作为项目的稳定版本，不再直接上传个人工作内容。
- 每位组员都必须建立自己的个人分支。
- 每个人只在自己的分支中添加、修改和提交文件。
- 工作完成后，通过 GitHub Pull Request 请求合并到 `main`。
- 以前已经上传到 `main` 的文件继续保留，不需要删除，也不需要重新上传。
- 每个人都应当从最新的 `main` 分支创建自己的个人分支。

> **重要提醒**
>
> 修改文件前，请先检查 VS Code 左下角显示的是不是自己的个人分支。  
> 如果显示的是 `main`，请不要继续修改、提交或推送个人工作内容。

## 1. 什么是 `main` 分支和个人分支

可以把 GitHub 仓库理解成一个共享项目文件夹。分支就是这个项目的不同工作线。

- `main` 是全组共享的正式工作区，应该保持稳定。
- 个人分支是每位组员自己的独立工作区。
- 在个人分支修改文件，不会立即影响其他组员。
- Pull Request 审核并合并后，个人工作才会进入 `main`。

示意图：

```text
main
├── zihan-chai
├── member-a
├── member-b
├── member-c
└── member-d
```

上面的图只是帮助理解。实际 Git 分支是平行的工作线，不是真正位于 `main` 文件夹下面，也不是 `main` 里面的子文件夹。

## 2. 个人分支命名规范

个人分支统一使用英文小写姓名，单词之间使用短横线。

推荐示例：

```text
zihan-chai
zhang-san
li-si
wang-wu
```

不要使用以下名称：

```text
new
test
branch1
mybranch
个人分支
```

如果组员主要负责某个任务，也可以在姓名后面加任务名称：

```text
zihan-chai-groundball
zhang-san-flyball
```

建议同一位组员长期使用固定分支，不要每次随意创建新名称。这样项目负责人更容易检查每个人的工作。

## 3. 第一次下载项目

第一次参与项目时，需要先把 GitHub 仓库下载到自己的电脑。

开始前请确认：

- 已经安装 Git。
- 已经安装 VS Code。
- 已经在 VS Code 中登录自己的 GitHub 账号。

### 方法一：使用 VS Code 图形界面

1. 打开 VS Code。
2. 按下 `Ctrl + Shift + P`。
3. 输入并选择 `Git: Clone`。
4. 粘贴 GitHub 仓库地址，例如：

   ```text
   https://github.com/REVER1E-YCQ/baseball-multimodal-dataset.git
   ```

5. 选择本地保存位置。
6. 下载完成后，VS Code 会询问是否打开项目，点击“打开”。
7. 打开后，确认左侧文件列表中可以看到 `dataset/`、`docs/` 和 `README.md`。

### 方法二：使用终端命令

在终端中运行：

```bash
git clone https://github.com/REVER1E-YCQ/baseball-multimodal-dataset.git
cd baseball-multimodal-dataset
```

如果使用其他仓库地址，请把命令中的地址替换成实际地址。

## 4. 从最新的 `main` 创建个人分支

这是最重要的一步。个人分支必须从最新的 `main` 创建，避免缺少其他组员已经上传的最新内容。

### 命令行方法

先切换到 `main`：

```bash
git switch main
```

拉取远程 `main` 的最新内容：

```bash
git pull origin main
```

创建并切换到自己的个人分支。下面用 `zihan-chai` 举例，实际使用时请替换成自己的分支名称：

```bash
git switch -c zihan-chai
```

第一次把个人分支推送到 GitHub：

```bash
git push -u origin zihan-chai
```

命令解释：

- `git switch main`：切换到主分支。
- `git pull origin main`：获取主分支最新内容。
- `git switch -c zihan-chai`：创建并切换到个人分支。
- `git push -u origin zihan-chai`：把个人分支上传到 GitHub，并记录之后默认推送到哪里。

> **请替换分支名称**
>
> 文档中的 `zihan-chai` 只是示例。每位组员都要替换成自己的实际分支名称。

## 5. 使用 VS Code 创建个人分支

如果不熟悉命令行，推荐优先使用 VS Code 图形界面。

1. 打开项目文件夹。
2. 查看 VS Code 窗口左下角的当前分支名称。
3. 如果当前不是 `main`，点击左下角分支名称，先切换到 `main`。
4. 点击左侧“源代码管理”图标。
5. 点击同步或拉取按钮，先获取远程 `main` 的最新内容。
6. 再次点击左下角的 `main`。
7. 选择“创建新分支”或 `Create new branch`。
8. 输入个人分支名称，例如 `zihan-chai`。
9. 确认 VS Code 左下角显示的分支已经变成 `zihan-chai`。
10. 点击左侧“源代码管理”图标。
11. 点击“发布分支”或 `Publish Branch`，把分支推送到 GitHub。

> **修改文件前必须检查**
>
> 修改文件前，先检查 VS Code 左下角显示的是不是自己的分支。  
> 如果显示的是 `main`，不要继续修改和提交。

## 6. 在个人分支中添加和修改文件

每位组员只应该在自己的数据目录中工作，不要修改其他组员的文件夹。

例如，Zihan Chai 负责地滚球数据时，可以在类似下面的目录中工作：

```text
dataset/
└── ground_ball/
    └── Zihan_Chai/
        └── G_001/
            ├── audio.wav
            ├── label.txt
            ├── sample.csv
            └── source.txt
```

如果负责高飞球数据，可以在类似下面的目录中工作：

```text
dataset/
└── fly_ball/
    └── Zihan_Chai/
        └── F_001/
            ├── audio.wav
            ├── label.txt
            ├── sample.csv
            └── source.txt
```

上传前请检查：

- 文件是否放在自己的文件夹中。
- 文件名、标签文件和来源信息是否完整。
- WAV、CSV、TXT 等文件是否可以正常打开。
- 不要修改其他组员的文件夹。
- 不要随意修改项目公共模板。
- 不要删除其他人的数据。
- 不要上传临时文件、缓存文件或没有完成的测试文件。

## 7. 使用 VS Code 提交并推送文件

完成文件修改后，需要提交并推送到自己的个人分支。

### VS Code 图形界面步骤

1. 完成文件修改。
2. 再次确认 VS Code 左下角显示的是自己的个人分支。
3. 点击 VS Code 左侧“源代码管理”图标。
4. 检查 `Changes` 中列出的文件。
5. 确认这些文件都是自己本次要提交的内容。
6. 点击文件旁边的 `+`，将文件加入暂存区。
7. 在消息框填写清楚的提交说明。
8. 点击“提交”或 `Commit`。
9. 点击“同步更改”或 `Push`，把提交推送到 GitHub。

提交信息应该清楚说明本次工作内容。

推荐示例：

```text
Add Zihan Chai groundball sample G_001
```

```text
Update flyball annotation for F_003
```

```text
Fix source information for G_002
```

不要使用无意义提交信息：

```text
update
111
修改
test
```

### 对应命令

```bash
git status
git add .
git commit -m "Add Zihan Chai groundball sample G_001"
git push
```

`git status` 用来检查当前分支和文件状态。提交前建议每次都运行或在 VS Code 中仔细查看 `Changes` 列表。

## 8. 每次开始工作前同步 `main`

每次开始新一轮工作前，个人分支都应该先同步 `main` 的最新内容。这样可以减少重复工作和冲突。

推荐初学者使用下面的命令：

```bash
git switch main
git pull origin main
git switch zihan-chai
git merge main
```

请把 `zihan-chai` 替换成自己的个人分支名称。

也可以使用：

```bash
git fetch origin
git merge origin/main
```

对初学者主要推荐第一种方法，因为步骤更直观：先更新 `main`，再回到自己的分支，把最新 `main` 合并进来。

如果出现冲突，不要随意删除冲突内容。请保存错误信息或截图，并联系项目负责人一起处理。

## 9. 在 GitHub 创建 Pull Request

个人分支中的工作完成后，不要直接合并到 `main`。应该创建 Pull Request，让项目负责人检查后再合并。

1. 确认个人分支已经推送到 GitHub。
2. 打开 GitHub 仓库网页。
3. 如果页面顶部出现 `Compare & pull request`，点击它。
4. 如果没有出现，可以点击 `Pull requests`，再点击 `New pull request`。
5. 检查分支设置：
   - base 分支为 `main`
   - compare 分支为自己的个人分支
6. 填写 Pull Request 标题和说明。
7. 点击 `Create pull request`。
8. 等待项目负责人检查。
9. 未经负责人确认，不要自行点击合并。

Pull Request 标题示例：

```text
Add Zihan Chai groundball samples
```

Pull Request 描述模板：

```markdown
## 本次提交内容

- 新增地滚球样本：
  - G_001_ground_ball
  - G_002_ground_ball

## 已检查内容

- [x] WAV 文件可以正常播放
- [x] CSV 标签格式正确
- [x] source.txt 包含视频标题和视频网址
- [x] 文件存放在自己的文件夹中
- [x] 没有修改其他组员的文件
```

## 10. 已经直接上传到 `main` 的组员怎么办

以前已经直接上传到 `main` 的内容不需要重新处理。

- 以前已经上传到 `main` 的文件继续保留。
- 不要删除已有文件。
- 不需要把已有文件重新上传一遍。
- 只需要从最新的 `main` 创建自己的个人分支。
- 从下一次工作开始，在个人分支中修改和提交。

如果已经在 `main` 中做了尚未提交的修改，请先不要提交到 `main`。可以使用下面的方法创建新分支：

```bash
git switch -c 自己的分支名称
git add .
git commit -m "提交说明"
git push -u origin 自己的分支名称
```

未提交的修改通常会一起带到新分支。因此可以先切换到个人分支，再提交这些修改。

如果已经在 `main` 中提交了内容，但还没有推送，请联系项目负责人协助处理，不要自己强制回退。

## 11. 如何确认自己当前在哪个分支

### VS Code 方法

查看 VS Code 窗口左下角的分支名称。那里显示的就是当前分支。

### 命令行方法

```bash
git branch
```

或：

```bash
git status
```

`git branch` 示例：

```text
* zihan-chai
  main
```

星号 `*` 表示当前所在分支。上面的例子表示当前在 `zihan-chai` 分支。

## 12. 常见错误及解决方法

### 错误一：不小心在 `main` 中修改了文件，但还没有提交

解决方法：

```bash
git switch -c 自己的分支名称
```

然后在个人分支中提交：

```bash
git add .
git commit -m "清楚描述本次工作"
git push -u origin 自己的分支名称
```

### 错误二：在 `main` 中已经提交，但还没有推送

先创建个人分支保存提交，然后让项目负责人协助处理 `main`。

不要自己强制回退，不要运行不理解的清理命令。

### 错误三：`git push` 提示没有上游分支

说明这个个人分支还没有和 GitHub 上的远程分支建立默认对应关系。

解决方法：

```bash
git push -u origin 自己的分支名称
```

之后再推送时，通常只需要运行：

```bash
git push
```

### 错误四：切换分支时提示有未提交修改

解决方法：

- 先确认这些修改是否属于自己。
- 如果属于自己，可以创建个人分支后提交。
- 如果不确定，请先截图并联系项目负责人。
- 不要直接使用强制删除或清理命令。

### 错误五：出现代码或文件冲突

解决方法：

- 不要直接删除带有冲突标记的内容。
- 不要使用 `git push --force`。
- 保存错误信息或截图。
- 联系项目负责人共同处理。

冲突标记通常长这样：

```text
<<<<<<< HEAD
当前分支的内容
=======
另一个分支的内容
>>>>>>> main
```

看到这种内容时，请不要随意删除。

### 错误六：提交到了其他组员的分支

解决方法：

- 立即停止继续提交。
- 不要强制推送。
- 联系项目负责人。
- 切换回自己的分支后再继续工作。

## 13. 严格禁止的操作

> **警告：普通组员未经负责人指导，不得执行以下命令**
>
> ```bash
> git push --force
> git reset --hard
> git clean -fd
> ```

这些命令可能覆盖远程内容、删除本地修改或造成数据丢失。

同时请严格遵守：

- 不要直接向 `main` 推送个人文件。
- 不要删除其他组员的数据。
- 不要修改其他组员的分支。
- 不要把整个项目重新复制一份放入仓库。
- 不要上传 `.vscode` 缓存、系统临时文件等无关内容。
- 不要把 GitHub 密码、Token 或其他账号信息写进仓库。

## 14. 组员日常操作流程

每次开始工作时，可以按下面流程操作。

```text
1. 打开 VS Code
2. 检查当前分支
3. 切换到 main
4. 拉取最新 main
5. 切换回个人分支
6. 合并 main 的最新内容
7. 在自己的文件夹中工作
8. 检查文件
9. Commit
10. Push
11. 创建 Pull Request
12. 等待负责人审核
```

命令版流程：

```bash
git switch main
git pull origin main
git switch 自己的分支名称
git merge main

# 添加或修改文件后
git status
git add .
git commit -m "清楚描述本次工作"
git push
```

最后再次提醒：命令中的 `自己的分支名称`、`zihan-chai`、提交说明等内容都要替换成自己的实际信息。不要把示例文字原样复制后直接使用。
