# Skills Roadmap — 復刻課程的 Agentic Flow

> 課程 Chapter 5 用 9 個 skill 把 Cosmos AnomalyGen 管線拆成可被 agent 呼叫的模組
> （簡報 slide 10）。我們用 **Claude Code 的專案級 skills** 復刻同一套 agentic flow。
> 這是本專案最容易做出差異化的部分：不只是「跑出合成資料」，而是「把整條 SDG 管線
> 做成可被自然語言驅動的工作流」。

**建置原則：一邊實作一邊建 skill，不預先寫空殼。** 每個 skill 在它負責的里程碑完成、
流程真的跑通之後才寫——skill 的內容必須是「已驗證可行的 SOP」，不是想像中的步驟。

---

## 對照表

| 課程 skill | 我們的 skill | 職責 | 建置時機 |
|---|---|---|---|
| `anomalygen`（orchestrator） | **`defectforge`** ✅ | 脈絡恢復與階段路由：讀 CLAUDE.md / PLAN.md / worklog / decisions，回報「現在在哪、上次做到哪、下一步」，並導向對應階段 skill | **M0（已建）** |
| `setup` | `df-setup` | 建 uv 環境、下載 VisA 並校驗 SHA256、檢查磁碟與 GPU、確認所有 checkpoint 就位 | M1–M2 |
| — | `df-split` | 跑 spot-diff、pHash 分群、凍結 split manifest、建 test blocklist、產 few-shot contact sheet | M3–M5 |
| — | `df-types` | 瑕疵分型（DINOv2 + 形態特徵分群）、產每群 contact sheet、等使用者確認命名、凍結 `defect_types.json` | M6 |
| `anomalygen-guard` | `df-guard` | Preflight 護欄：split 是否已凍結、manifest SHA256 是否相符、任何輸入路徑是否命中 test blocklist、磁碟／VRAM 是否足夠、有無明文金鑰。**任何生成或訓練動作前必跑** | M4 之後（有東西可 guard 才建） |
| — | `df-stage-a` | Stage A 合成：copy-paste 與程序化 anomaly，含自動斷言與抽樣 grid 檢視 | M7–M8 |
| `prep-testcase` | `df-prep-testcase` | Auto mask placement，並依真實 mask 元件數**按比例分配 `num_SDG`**，輸出 SDG-ready 的 (clean, mask) 清單 | M9 |
| `finetune` | `df-finetune` | 驗證訓練資料、產訓練 config、跑本機 smoke test、把 notebook 交接給使用者上 Colab | M10–M11 |
| `sdg-inference` | `df-sdg` | 從指定 LoRA checkpoint 批次生成到 `original/` 桶，支援 `--resume` | M12 |
| `sdg-refine` | `df-refine` | 對 `original/` 做 `(guidance_scale, crop_ratio)` 逐樣本搜尋，組裝最佳結果進 `searched/` 桶 | M12 |
| — | `df-filter` | 六道過濾、輸出 filtered/unfiltered 兩版、產漏斗報表 | M13 |
| `eval` | `df-eval` | per-type `nn_score`（主 KPI）／`mnn_score`／FID／KID，含健全性檢查 | M14 |
| `anomalygen-release` | `df-release` | 發佈前總驗收：重現性、數字誠實性、防洩漏自查、授權標註、HF card、金鑰掃描 | Phase 2 收尾 |

課程另有 `anomalygen-release` 的 Docker 建置職責，我們的 `df-release` 改為對應
「發佈到 GitHub / Hugging Face 前的完整檢查」，因為本專案沒有容器化需求。

---

## Skill 撰寫規範

放在 `.claude/skills/<name>/SKILL.md`，frontmatter 至少含：

```yaml
---
name: <skill-name>
description: <一句話，寫清楚「什麼情況該用它」，因為這是觸發判斷的依據>
---
```

內容要求：

1. **先寫前置條件**（哪些里程碑必須已完成、哪些檔案必須存在），不滿足就中止並回報
2. **每一步都要有可判定成敗的驗證**，不要只寫「執行 X」
3. **明確寫出停下來問使用者的時機**（例如 `df-types` 必須等使用者確認型別名稱）
4. **對應到 PLAN.md 的里程碑編號**，方便回頭追
5. 完成後要求：更新 PLAN.md 勾選 → 追加 worklog → git commit → 給「換你做」清單

---

## 為什麼這件事值得做

課程 slide 10 對 agentic flow 的定義是
「Observe → Plan → Call tools → Check results → Re-plan」，
以及「從『給你一段文字回覆』變成『幫你跑完一個工作流』」。

一個只有 notebook 的 repo 展示的是「我會訓模型」；
一個把管線拆成 13 個可被自然語言驅動、各自帶驗證與護欄的 skill 的 repo，
展示的是「我會設計可被 agent 操作的資料生產系統」——後者才是這個作品集的差異點。
