/** 字符级分词器(小词表): 字 ↔ 编码 */
export class CharTokenizer {
  private chars: string[] = [];
  private map = new Map<string, number>();

  fit(text: string): void {
    const seen = new Set<string>();
    for (const ch of text) seen.add(ch);
    this.chars = [...seen].sort((a, b) => a.codePointAt(0)! - b.codePointAt(0)!);
    this.map.clear();
    this.chars.forEach((ch, i) => this.map.set(ch, i));
  }

  /** 按出现频率截断词表,低频字映射到 OOV(id=词表长度) */
  fitTopN(text: string, maxChars: number): void {
    const freq = new Map<string, number>();
    for (const ch of text) freq.set(ch, (freq.get(ch) ?? 0) + 1);
    const chars = [...freq.entries()]
      .sort((a, b) => b[1] - a[1])
      .slice(0, maxChars - 1)
      .map(([ch]) => ch);
    chars.push("\uFFFD"); // 末位保留 UNK 槽
    this.load(chars);
  }

  load(chars: string[]): void {
    this.chars = [...chars];
    this.map.clear();
    this.chars.forEach((ch, i) => this.map.set(ch, i));
  }

  get vocabSize(): number {
    return this.chars.length;
  }

  idOf(ch: string): number | undefined {
    return this.map.get(ch);
  }

  charOf(id: number): string {
    return this.chars[id] ?? "?";
  }

  encode(text: string): number[] {
    const out: number[] = [];
    const unk = this.chars.length - 1;
    for (const ch of text) {
      const id = this.map.get(ch);
      out.push(id ?? unk);
    }
    return out;
  }

  decode(ids: number[]): string {
    return ids.map((i) => this.chars[i] ?? "?").join("");
  }

  charset(): string[] {
    return [...this.chars];
  }
}
