# ============================================================
# 天玄 TianXuan · Dockerfile
# 用法:
#   docker build -t tianxuan .
#   docker run -it --rm -p 3800:3800 tianxuan          # Web 服务
#   docker run -it --rm tianxuan train                   # 训练
#   docker run -it --rm tianxuan generate --prompt "小明" # 生成
# ============================================================
FROM node:22-slim

LABEL maintainer="TianXuan"
LABEL description="天玄 · 多神经协同字符级小语言模型"

# 系统依赖: curl(语料下载), python3(json解析/检查点读取)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl python3 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 拷贝项目文件
COPY package.json package-lock.json ./
COPY tsconfig.json ./
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY AGENTS.md PLAN.md README.md ./

# 安装依赖 + 编译
RUN npm install --no-audit --no-fund && npx tsc --outDir dist

# 拉取语料
RUN bash scripts/fetch-zh-classics.sh data/raw

# 生成训练语料(500万字符, 与训练参数一致)
RUN node -e " \
  const { buildCorpus } = require('./dist/ml/data'); \
  const fs = require('fs'); \
  fs.mkdirSync('data', { recursive: true }); \
  const b = buildCorpus({ seed: 7, tokens: 5000000, rawPerFile: 30000 }); \
  fs.writeFileSync('data/corpus.txt', b.text, 'utf-8'); \
  console.log('Corpus: ' + b.text.length + ' chars'); \
"

# 默认端口
EXPOSE 3800

# 入口: 根据 CMD 参数决定行为
ENTRYPOINT ["node"]
CMD ["dist/server/index.js"]
