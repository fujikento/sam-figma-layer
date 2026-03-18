# SAM-Figma-Layer セットアップガイド

## 📋 完全自動セットアップ手順

### Step 1: 初期セットアップ（10分）

```bash
# 1. ディレクトリ移動
cd ~/.openclaw/workspace/mcp-servers/sam-figma-layer

# 2. Python依存関係 + SAMモデルダウンロード（2.4GB、初回のみ）
npm run setup

# 待機中の表示:
# 📦 Installing Python dependencies...
#    Installing segment-anything...
#    Installing opencv-python...
#    Installing numpy...
#    Installing pillow...
# ✅ Dependencies installed
#
# 🤖 Downloading SAM model (2.4GB)...
#    [████████████████████████████████████████] 100.0%
# ✅ Model downloaded
#
# 🔍 Verifying installation...
# ✅ All Python packages imported successfully
# ✅ Setup completed successfully!

# 3. Node.js依存関係インストール
npm install

# 4. TypeScriptビルド
npm run build
```

---

### Step 2: Figma Token取得（3分）

#### 1. Figmaにログイン
https://www.figma.com/

#### 2. Settings → Personal Access Tokens
https://www.figma.com/developers/api#access-tokens

#### 3. Generate new token
- Token name: `Claude Code SAM Integration`
- Expiration: `Never` or `90 days`
- Scopes: `File content` (Read & Write)

#### 4. Tokenをコピー → 環境変数にセット

**macOS/Linux:**
```bash
# ~/.zshrc または ~/.bashrc に追加
export FIGMA_ACCESS_TOKEN='figd_XXXXXXXXXXXXXXXXXXXXXXXXXXXX'

# 反映
source ~/.zshrc
```

**確認:**
```bash
echo $FIGMA_ACCESS_TOKEN
# → figd_XXXXXXXXXXXXXXXXXXXXXXXXXXXX
```

---

### Step 3: Claude Codeに登録（2分）

#### 方法A: 自動登録（推奨）
```bash
claude mcp add sam-figma-layer -- node ~/.openclaw/workspace/mcp-servers/sam-figma-layer/dist/index.js
```

#### 方法B: 手動設定
```bash
# 設定ファイルを開く
nano ~/.claude/claude_desktop_config.json
```

以下を追加:
```json
{
  "mcpServers": {
    "sam-figma-layer": {
      "command": "node",
      "args": [
        "/Users/kento/.openclaw/workspace/mcp-servers/sam-figma-layer/dist/index.js"
      ],
      "env": {
        "FIGMA_ACCESS_TOKEN": "figd_XXXXXXXXXXXXXXXXXXXXXXXXXXXX"
      }
    }
  }
}
```

保存して終了: `Ctrl+O` → `Enter` → `Ctrl+X`

---

### Step 4: 動作確認（1分）

#### 1. Claude Code / Desktopを再起動

#### 2. Claude Codeで確認
```
User: 「利用可能なツールをリストして」
```

以下が表示されればOK:
- ✅ `segment_image`
- ✅ `segment_and_upload_to_figma`
- ✅ `check_sam_status`

#### 3. SAMモデル確認
```
User: 「SAMの状態を確認して」
```

表示:
```
✅ SAM model is ready
```

---

## 🎯 使い方

### 基本: 画像セグメンテーション

```
User: 「~/Downloads/test.pngをSAMで分離して」

Claude: [自動実行]
segment_image({
  imagePath: "/Users/kento/Downloads/test.png"
})
```

**結果例:**
```
✅ Segmentation complete!

📊 Results:
- Total layers: 18
- Background: Layer 0
- Foreground objects: 5
- Small elements: 12

📁 Output directory: /tmp/sam_output_1234567890

Layers:
- Layer 0: layer_000.png (1024x768)
- Layer 1: layer_001.png (512x384)
- Layer 2: layer_002.png (256x192)
...
```

---

### 完全自動: 分離 → Figma

```
User: 「~/Downloads/hero.pngをSAMで分離して、
       Figmaファイル ABC123XYZ にアップロードして」

Claude: [自動実行]
segment_and_upload_to_figma({
  imagePath: "/Users/kento/Downloads/hero.png",
  figmaFileKey: "ABC123XYZ"
})
```

**Figma File Key取得方法:**
```
Figma URL: https://www.figma.com/design/ABC123XYZ/Project-Name
                                     ↑↑↑↑↑↑↑↑↑
                                     この部分
```

**結果例:**
```
✅ Complete! Segmented and uploaded to Figma

📊 SAM Segmentation:
- Total layers: 15
- Background: Layer 0
- Foreground: 4 objects
- Elements: 10 small elements

🎨 Figma Upload:
- File: https://figma.com/file/ABC123XYZ
- Uploaded layers: 15

Layers created:
- Background (Layer 0)
- Object_1 (Layer 1)
- Object_2 (Layer 2)
- Object_3 (Layer 3)
- Object_4 (Layer 4)
- Element_5 (Layer 5)
...
```

---

## 🔧 トラブルシューティング

### 1. "SAM model not found" エラー

```bash
# 再度セットアップ実行
cd ~/.openclaw/workspace/mcp-servers/sam-figma-layer
npm run setup
```

### 2. Python依存関係エラー

```bash
# 手動インストール
pip3 install segment-anything opencv-python numpy pillow
```

### 3. "Figma token not provided" エラー

```bash
# 環境変数確認
echo $FIGMA_ACCESS_TOKEN

# 設定されてなければ追加
export FIGMA_ACCESS_TOKEN='figd_XXXXXXXXXXXXXXXXXXXXXXXXXXXX'
source ~/.zshrc
```

### 4. Claude CodeがツールR認識しない

```bash
# Claude Code完全再起動
# macOS: Cmd+Q で終了 → 再起動

# 設定確認
cat ~/.claude/claude_desktop_config.json | jq .mcpServers
```

### 5. SAM処理が遅い

**原因:** CPUで処理している（GPUなし）

**対策:**
- **M4チップ**: すでに最適化済み（問題なし）
- **処理時間目安**:
  - 1024x1024: 30-60秒
  - 2048x2048: 2-3分
  - 4096x4096: 5-10分

---

## 💡 実用例

### 例1: DAIMASU AIRLINESヒーロー

```
User: 「Geminiで生成した ~/daimasu-hero.png を
       Figmaファイル FGA456DEF に分離アップロードして」
```

### 例2: Lobster Pokerキャラクター

```
User: 「~/lobster-char.png をSAMで分離。
       ザリガニ本体と背景を別レイヤーにしたい」
```

### 例3: B-Ticketモバイル画面

```
User: 「~/bticket-screens/ の全PNG画像を
       一括でSAM分離してFigmaにアップロード」
       
Claude: [バッチ処理実行]
```

---

## 📊 パフォーマンス

| 画像サイズ | レイヤー数 | 処理時間 | 出力サイズ |
|---|---|---|---|
| 1024x768 | 10-20 | 30-60秒 | 5-10MB |
| 2048x1536 | 20-40 | 2-3分 | 20-40MB |
| 4096x3072 | 40-80 | 5-10分 | 80-150MB |

---

## 🚀 次のステップ

### 1. テスト実行
```bash
# テスト画像でSAM動作確認
# (任意の画像で試す)
```

### 2. Gemini統合
```bash
# nano-banana-proスキルと統合
# Gemini生成 → SAM分離 → Figmaアップロード
```

### 3. バッチ処理
```bash
# 複数画像の一括処理機能追加
```

---

**セットアップ完了したら「完了」と報告してください！**
