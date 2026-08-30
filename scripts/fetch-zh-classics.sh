#!/usr/bin/env bash
# 天玄 TianXuan · 公版中文语料拉取脚本(Project Gutenberg, UTF-8)
# 用法: bash scripts/fetch-zh-classics.sh [输出目录]
# 数据来源: https://www.gutenberg.org/browse/languages/zh (公有领域)
set -euo pipefail

OUT="${1:-data/raw}"
mkdir -p "$OUT"

# 编号 | 书名(Gutenberg 中文书目; 注: 部分为选本/删节)
BOOKS=(
  "23950|三国志演义"
  "23863|水浒传"
  "24264|红楼梦"
  "23962|西游记"
  "24032|儒林外史"
  "51828|聊斋志异"
  "24226|史记"
  "25349|东周列国志"
  "23835|隋唐演义"
  "23910|封神演义"
  "24047|世说新语"
)

for entry in "${BOOKS[@]}"; do
  pid="${entry%%|*}"
  name="${entry##*|}"
  if [ -s "$OUT/${name}.txt" ]; then
    echo "==> 跳过 ${name} (已存在)"
    continue
  fi
  url="https://www.gutenberg.org/cache/epub/${pid}/pg${pid}.txt"
  tmp="$OUT/.${name}.tmp"
  echo "==> 拉取 ${name} (pg${pid})"
  if curl -fsSL --retry 5 --retry-delay 2 --max-time 180 \
      -A "TianXuanCorpus/0.1 (data pipeline)" "$url" -o "$tmp"; then
    if head -c 200 "$tmp" | iconv -f utf-8 -t utf-8 >/dev/null 2>&1; then
      size=$(wc -c < "$tmp")
      if [ "$size" -lt 100000 ]; then
        echo "    [跳过] 文件过小($size bytes),疑似截断"
        rm -f "$tmp"
      else
        mv "$tmp" "$OUT/${name}.txt"
        echo "    已保存: $OUT/${name}.txt ($size bytes)"
      fi
    else
      echo "    [跳过] 非 UTF-8 编码,请人工处理: $tmp"
    fi
  else
    echo "    [失败] 下载失败 pg${pid}"
    rm -f "$tmp"
  fi
done
echo "完成。用 npm run train -- --tokens N --rawperfile 200000 即可接入新语料。"
