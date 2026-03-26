# Data Pipeline Guide — stock-ai-agent

## 1. Tổng Quan Architecture

### Sơ Đồ Luồng Dữ Liệu

```
[Prefect Scheduler] ── cron triggers ──> [FastAPI Internal Endpoints]
    │
    │  ┌─ news-crawl flow (3x/ngày) ─────────────────────────┐
    ├──│ 1. POST /internal/trigger/crawl                      │
    │  │    └── News Crawlers (Vietstock, CafeF, VnEconomy)   │
    │  │        ↓ RSS feeds → parse HTML → extract tickers     │
    │  │        ↓ Category tagging: "stock"/"macro" từ config  │
    │  │        [articles table] ── dedup by URL               │
    │  │ 2. POST /internal/trigger/embedding                   │
    │  │    └── OpenAI text-embedding-3-small (1536-dim)       │
    │  │        [articles.embedding] ── pgvector HNSW index    │
    │  └──────────────────────────────────────────────────────-┘
    │
    │  ┌─ stock-pipeline flow (weekdays 17:00 VNT) ──────────┐
    ├──│ 1. POST /internal/trigger/stock-crawl                │
    │  │    └── vnstock Client → OHLCV cho ~44 tickers         │
    │  │        [market_data table] ── data_type="stock_price" │
    │  │ 2. POST /internal/trigger/technical-indicators        │
    │  │    └── Indicator Calculator (pandas-ta) → 10 indicators│
    │  │        [market_data table] ── data_type="technical_indicator"│
    │  └───────────────────────────────────────────────────────┘
    │
    │  ┌─ data-cleanup flow (daily 02:00 VNT) ───────────────┐
    └──│ 1. POST /internal/trigger/lifecycle                  │
       │    └── Lifecycle Pipeline                             │
       │        ├── Summary (gpt-4o-mini) → articles.summary   │
       │        └── Clear raw_content (>30 ngày)               │
       └───────────────────────────────────────────────────────┘
```

### Bảng Tổng Hợp Các Crawler

| Crawler | Source | Flow | Rate Limit | Output Table | Ghi Chú |
|---------|--------|------|------------|--------------|---------|
| Vietstock | 5 RSS feeds | news-crawl | 1 RPS | articles | Dedup by URL, category tagging |
| CafeF | 5 RSS feeds | news-crawl | 1 RPS | articles | Dedup by URL, category tagging |
| VnEconomy | 4 RSS feeds | news-crawl | 1 RPS | articles | Dedup by URL, category tagging |
| Embedding | OpenAI API | news-crawl | Batch 100 | articles.embedding | 1536-dim vector, chạy sau crawl |
| Stock Prices | vnstock API (VCI) | stock-pipeline | 1s giữa tickers | market_data | ~44 tickers |
| Tech Indicators | pandas-ta calc | stock-pipeline | 0.1s giữa tickers | market_data | 10 indicators |
| Lifecycle | gpt-4o-mini | data-cleanup | Batch 50 | articles.summary | >30 ngày |

### Database Schema Tổng Quan

- **articles** — Lưu bài viết tin tức tài chính (Vietstock, CafeF, VnEconomy), column `category` phân biệt "stock" và "macro"
- **market_data** — Lưu giá cổ phiếu, technical indicators

---

## 2. News Crawlers

### 2.1 Vietstock

**Source:** `services/crawler/news/vietstock_crawler.py`

**RSS Feeds (5 feeds, category tagging):**
- `vi-mo.rss` — category: **macro**
- `kinh-te-dau-tu.rss` — category: **macro**
- `co-phieu.rss` — category: **stock**
- `tai-chinh-quoc-te.rss` — category: **macro**
- `ngan-hang.rss` — category: **macro**

**Schedule:** Flow `news-crawl` — 3 lần/ngày lúc 06:00, 12:00, 18:00 VNT (`0 23,5,11 * * *` UTC)

**Cách hoạt động:**
1. Fetch RSS XML từ 5 feeds (mỗi feed có category config)
2. Parse mỗi `<item>`: lấy title, link, pubDate, description
3. Kiểm tra robots.txt trước khi fetch full article
4. Fetch HTML article page qua HTTP client (timeout 30s)
5. Parse nội dung bằng CSS selectors theo thứ tự ưu tiên:
   - `div.content-detail` → `div.article-content` → `div#content-detail` → `<article>` → fallback (div có nhiều `<p>` nhất)
6. Extract ticker symbols từ text (pattern: 3 chữ cái viết hoa `[A-Z]{3}`, loại trừ false positives như USD, VND, GDP, CPI...)
7. Dedup bằng URL — bài đã tồn tại sẽ bị skip
8. Gán `category` từ feed config ("macro" hoặc "stock") vào `ArticleCreate`

**Output:** Bảng `articles` — columns: source="vietstock", ticker_symbol, title, url (unique), published_at, raw_content, category

### 2.2 CafeF

**Source:** `services/crawler/news/cafef_crawler.py`

**RSS Feeds (5 feeds, category tagging):**
- `tai-chinh-quoc-te.rss` — category: **macro**
- `thi-truong-chung-khoan.rss` — category: **stock**
- `tai-chinh-ngan-hang.rss` — category: **macro**
- `vi-mo-dau-tu.rss` — category: **macro**
- `thi-truong.rss` — category: **stock**

**Schedule:** Flow `news-crawl` — cùng lịch với Vietstock

**Cách hoạt động:** Tương tự Vietstock, chỉ khác CSS selectors:
- `div.detail-content` → `div#mainContent` → `div.contentdetail` → `<article>` → fallback

**Output:** Bảng `articles` — source="cafef", category từ feed config

### 2.3 VnEconomy

**Source:** `services/crawler/news/vneconomy_crawler.py`

**RSS Feeds (4 feeds, category tagging):**
- `chung-khoan.rss` — category: **stock**
- `thi-truong.rss` — category: **stock**
- `tai-chinh.rss` — category: **macro**
- `kinh-te-the-gioi.rss` — category: **macro**

**Schedule:** Flow `news-crawl` — cùng lịch với Vietstock, CafeF

**CSS selectors:**
- `div.detail__content` → `div.article-body` → `div.content-detail` → `<article>` → fallback

**Output:** Bảng `articles` — source="vneconomy", category từ feed config

### 2.4 Crawl Manager & Rate Limiting

**Source:** `services/crawler/news/crawl_manager.py`

**Cơ chế thực thi:**
1. Load config từ `config/crawlers/sources.yaml`
2. Chạy từng crawler **tuần tự** (vietstock → cafef → vneconomy)
3. Mỗi source có HTTP client riêng với `RateLimitedTransport`
4. Sau khi crawl xong tất cả sources, gọi `save_articles()` để lưu vào DB

**Rate Limiting (RateLimitedTransport):**
- Source: `services/crawler/middleware/rate_limiter.py`
- Mechanism: Per-domain rate limiting dùng `pyrate_limiter.Limiter`
- Default: 1 request/second/domain
- Cấu hình qua `rate_limit_rps` trong `sources.yaml`

**Robots.txt Compliance:**
- Source: `services/crawler/middleware/robots_checker.py`
- Kiểm tra robots.txt trước mỗi request
- Cache per-domain, TTL = 86400 giây (24 giờ)
- User-Agent: `StockAIAgent/1.0`
- Nếu fetch robots.txt fail → cho phép request (permissive)

**Deduplication (Article Repo):**
- Source: `services/crawler/news/article_repo.py`
- Batch query URLs đã tồn tại: `SELECT url FROM articles WHERE url IN (...)`
- Chỉ insert bài mới (URL chưa có trong DB)
- Constraint unique trên column `url`

**HTTP Client Config:**
- Timeout: 30 giây
- User-Agent: `StockAIAgent/1.0`

---

## 3. Macro News (Category-Based Approach)

Thay vì crawl số liệu vĩ mô từ APIs riêng (VN-Index, USD/VND, SBV...), hệ thống phân tích vĩ mô dựa trên **tin tức vĩ mô** được crawl qua news pipeline có sẵn.

### Cách Hoạt Động

1. RSS feeds trong `config/crawlers/sources.yaml` được gán `category: "macro"` hoặc `"stock"`
2. News crawlers (Vietstock, CafeF, VnEconomy) đọc category từ feed config
3. Khi tạo `ArticleCreate`, category được gán tự động từ feed config
4. Articles lưu vào bảng `articles` với column `category` (default: "stock")
5. Market Context Agent query `WHERE category = 'macro'` để lấy tin vĩ mô
6. Prompt `macro_analysis.yaml` v2.0 nhận danh sách bài báo thay vì số liệu

### Category Mapping

| Source | Feed Keywords | Category |
|--------|--------------|----------|
| Vietstock | vi-mo, kinh-te-dau-tu, tai-chinh-quoc-te, ngan-hang | macro |
| Vietstock | co-phieu | stock |
| CafeF | tai-chinh-quoc-te, tai-chinh-ngan-hang, vi-mo-dau-tu | macro |
| CafeF | thi-truong-chung-khoan, thi-truong | stock |
| VnEconomy | tai-chinh, kinh-te-the-gioi | macro |
| VnEconomy | chung-khoan, thi-truong | stock |

### Backward Compatibility

- Column `category` có `server_default="stock"` → articles cũ tự động gán "stock"
- Crawlers hỗ trợ backward-compatible: nếu feed config là string (không phải dict), default category = "stock"
- Lifecycle cleanup sẽ xóa raw_content sau 30 ngày → sau vài tuần toàn bộ articles mới sẽ có category chính xác

---

## 4. Stock History (vnstock Client)

### 4.1 Initial Load vs Incremental

**Source:** `services/crawler/market_data/stock_crawl_manager.py`

**Schedule:** Flow `stock-pipeline` — hàng ngày trading days lúc 17:00 VNT (`0 10 * * 1-5` UTC, sau khi thị trường đóng cửa)

**Logic phân loại:**
- Đếm rows hiện có cho mỗi ticker trong DB
- `INITIAL_THRESHOLD = 30` rows
- Ticker < 30 rows → **Initial load**: fetch `1Y` (1 năm) dữ liệu
- Ticker ≥ 30 rows → **Incremental load**: fetch `1b` (1 ngày giao dịch gần nhất)

**Ngày không giao dịch:**
- Chỉ chạy initial load cho tickers mới
- Skip incremental load (không có dữ liệu mới)

**Rate limit:** `asyncio.sleep(1.0)` — 1 giây giữa mỗi ticker

### 4.2 Ticker Configuration

**Source:** `services/crawler/market_data/ticker_config.py`
**Config:** `config/crawlers/stock_tickers.yaml`

**Các nhóm ticker:**

| Nhóm | Mô Tả | Số Lượng | Ví Dụ |
|------|--------|----------|-------|
| vn30 | VN30 Index components | 30 | VNM, VHM, VIC, HPG, FPT, VCB... |
| securities | Công ty chứng khoán | 5 | SSI, VND, HCM, VCI, SHS |
| oil_gas | Dầu khí | 5 | GAS, PLX, PVD, PVS, BSR |
| banking | Ngân hàng | 10 | VCB, BID, CTG, TCB, MBB... |
| steel | Thép | 4 | HPG, HSG, NKG, TLH |

**Deduplication:** Ticker xuất hiện trong nhiều nhóm (VD: HPG ở vn30 + steel, VCB ở vn30 + banking) sẽ được deduplicate → ~44 tickers duy nhất.

**Validation:** Pattern `^[A-Z]{3,4}$` — 3-4 chữ cái viết hoa. Ticker không hợp lệ sẽ bị skip với warning.

### 4.3 Trading Day Detection

**Source:** `services/crawler/market_data/stock_crawl_manager.py`

Hàm `is_trading_day()` kiểm tra:
1. **Cuối tuần:** Thứ 7, Chủ nhật → không giao dịch
2. **Ngày lễ cố định (hardcoded):**
   - 1/1 — Tết Dương lịch
   - 30/4 — Ngày Giải phóng
   - 1/5 — Quốc tế Lao động
   - 2/9 — Quốc khánh
3. **Ngày lễ thay đổi (từ config):** `config/crawlers/stock_tickers.yaml`
   - VD năm 2026: Tết Nguyên Đán (16-20/2), Giỗ Tổ Hùng Vương (6/4), Nghỉ bù 1/5 (3/5)

### 4.4 vnstock Client

**Source:** `services/crawler/market_data/vnstock_client.py`

**Quote source:** `VCI` (mặc định cho price data)
**Finance source:** `KBS` (mặc định cho BCTC)

**Retry config:** Exponential backoff (min 1s, max 10s), tối đa 3 lần

**Mock fallback:** Nếu API fail sau 3 lần retry → trả mock data với `data_source="mock"`, log warning

**Methods chính:**
- `aget_stock_history(ticker, length, interval)` — OHLCV data
- `aget_financial_ratios(ticker)` — P/E, P/B, ROE, EPS
- `aget_income_statement(ticker)` — Báo cáo thu nhập
- `aget_balance_sheet(ticker)` — Bảng cân đối kế toán

---

## 5. Technical Indicators Calculator

**Source:** `services/crawler/market_data/indicator_calculator.py`, `indicator_manager.py`

**Schedule:** Flow `stock-pipeline` — chạy sau stock-crawl trong cùng flow

### Flow Tính Toán

1. Load ticker config (~44 tickers)
2. Kiểm tra trading day — skip nếu không giao dịch
3. Fetch VN-Index OHLCV (300 rows gần nhất) cho tính RS
4. Với mỗi ticker:
   - Fetch OHLCV data (300 rows)
   - Tính 10 indicators bằng `pandas-ta`
   - Lưu vào `market_data` với `data_type="technical_indicator"`, `data_source="calculated"`
   - Sleep 0.1s (100ms) giữa tickers

### 10 Indicators

| Indicator | Mô Tả | Min Data |
|-----------|--------|----------|
| SMA_20 | Simple Moving Average 20 phiên | 20 |
| SMA_50 | Simple Moving Average 50 phiên | 50 |
| SMA_200 | Simple Moving Average 200 phiên | 200 |
| RSI_14 | Relative Strength Index (length=14) | 15 |
| MACD_LINE, MACD_SIGNAL, MACD_HISTOGRAM | MACD (fast=12, slow=26, signal=9) | 35 |
| VOLUME_AVG_20, VOLUME_CURRENT, VOLUME_RATIO | Volume Profile (MA 20) | 20 |
| ATR_14 | Average True Range (length=14) | 15 |
| BB_LOWER, BB_MIDDLE, BB_UPPER | Bollinger Bands (length=20, std=2) | 20 |
| DC_LOWER, DC_MIDDLE, DC_UPPER | Donchian Channels (length=20) | 20 |
| RS_VNINDEX | Relative Strength vs VN-Index (20 phiên) | 20 |

**Logic tính toán:**
- Chỉ lấy giá trị ngày gần nhất (`iloc[-1]`) từ OHLCV DataFrame
- Làm tròn 6 chữ số thập phân
- RS_VNINDEX: align dates giữa ticker và VN-Index, so sánh cùng ngày giao dịch
- Nếu data không đủ minimum periods → skip indicator đó, log warning

---

## 6. Article Embedding Pipeline

**Source:** `services/crawler/embedding/embedding_pipeline.py`, `shared/llm/embedder.py`

**Trigger:** `POST /internal/trigger/embedding` — chạy sau crawl trong flow `news-crawl`

### Cách Hoạt Động

1. Query articles có `embedded=FALSE`
2. Chuẩn bị text: `title + "\n\n" + raw_content` (hoặc summary nếu raw_content đã bị xóa)
3. Skip bài không có content (log warning)
4. Gọi OpenAI API embed theo batch:
   - **Model:** `text-embedding-3-small`
   - **Dimension:** 1536
   - **Batch size:** 100 (mặc định)
5. Update article: gán `embedding` vector + set `embedded=True`
6. Commit changes

**Retry config (OpenAI):** Exponential backoff (1-10s), 3 lần, retry trên RateLimitError/APITimeoutError/APIConnectionError

**pgvector Index:** HNSW (m=16, ef_construction=64), operator: `vector_cosine_ops`

---

## 7. Data Lifecycle Management

**Source:** `services/crawler/lifecycle/lifecycle_pipeline.py`

**Schedule:** Flow `data-cleanup` — hàng ngày lúc 02:00 VNT (`0 19 * * *` UTC)

### Cách Hoạt Động

1. Query articles có `published_at` > 30 ngày **VÀ** còn `raw_content`
2. Giới hạn tối đa 1000 bài/lần chạy
3. Xử lý theo batch 50 bài:
   - Skip nếu đã có summary
   - Gọi LLM tạo summary tiếng Việt
   - Lưu summary, **xóa raw_content** (set None)
   - **Giữ nguyên** embedding vector và flag embedded
4. Commit sau mỗi batch (rollback batch nếu fail, tiếp tục batch tiếp)

**LLM Config:**
- Model: `gpt-4o-mini`
- Temperature: 0.3
- Max tokens: 500
- Prompt template: `lifecycle/summarize_article`

**Kết quả:** Bài >30 ngày sẽ chỉ còn title, summary, embedding — raw_content bị xóa để tiết kiệm storage.

---

## 8. Prefect Scheduler

**Source:** `services/scheduler/flows/data_pipeline.py`
**Config:** `config/scheduler/schedules.yaml`

### 8.1 Schedule Overview

| Flow | Cron (UTC) | Giờ VN | Steps | Enabled |
|------|-----------|--------|-------|---------|
| news_crawl | `0 23,5,11 * * *` | 06:00, 12:00, 18:00 | crawl → embedding | true |
| stock_pipeline | `0 10 * * 1-5` | 17:00 (sau đóng cửa) | stock-crawl → technical-indicators | true |
| data_cleanup | `0 19 * * *` | 02:00 | lifecycle | true |
| morning_briefing | `0 0 * * 1-5` | 07:00 | Morning briefing | **false** |

### 8.2 Pipeline Flows

Hệ thống có 3 flow độc lập, mỗi flow chạy các bước **tuần tự**:

**news-crawl** — Crawl tin tức từ tất cả sources + embed bài viết:
```
crawl → embedding
```

**stock-pipeline** — Crawl giá cổ phiếu + tính technical indicators:
```
stock-crawl → technical-indicators
```

**data-cleanup** — Dọn dẹp dữ liệu cũ:
```
lifecycle
```

Mỗi bước gọi HTTP POST đến Internal API. Nếu 1 bước fail → log error → tiếp tục bước tiếp (không block pipeline).

### 8.3 Internal API Triggers

**Source:** `services/app/routers/internal.py`
**Prefix:** `/internal/trigger`
**Authentication:** Header `X-Trigger-Source: prefect-scheduler` (bắt buộc, trả 403 nếu thiếu)

| Endpoint | Method | Flow | Mô Tả |
|----------|--------|------|--------|
| `/internal/trigger/crawl` | POST | news-crawl | News crawl tất cả sources (including macro news) |
| `/internal/trigger/embedding` | POST | news-crawl | Article embedding |
| `/internal/trigger/stock-crawl` | POST | stock-pipeline | Stock history crawl |
| `/internal/trigger/technical-indicators` | POST | stock-pipeline | Technical indicator calculation |
| `/internal/trigger/lifecycle` | POST | data-cleanup | Data lifecycle cleanup |

**Kết nối:** Prefect scheduler gọi qua `APP_URL` (default: `http://app:8000`).

**News crawl (all sources)**
curl -X POST http://localhost:8000/internal/trigger/crawl -H "X-Trigger-Source: prefect-scheduler"

**Article embedding**
curl -X POST http://localhost:8000/internal/trigger/embedding -H "X-Trigger-Source: prefect-scheduler"

**Stock history crawl**
curl -X POST http://localhost:8000/internal/trigger/stock-crawl -H "X-Trigger-Source: prefect-scheduler"

**Technical indicator calculation**
curl -X POST http://localhost:8000/internal/trigger/technical-indicators -H "X-Trigger-Source: prefect-scheduler"

**Data lifecycle cleanup**
curl -X POST http://localhost:8000/internal/trigger/lifecycle -H "X-Trigger-Source: prefect-scheduler"

### 8.4 Monitoring

- **Prefect UI:** Dashboard theo dõi flow runs, task states, logs
- **Logs:** Structured JSON logging với component prefix (VD: `crawler.vietstock`, `crawler.cafef`)
- **Retry:** Mỗi pipeline step có error handling riêng, không retry ở scheduler level
- **Kết quả:** Mỗi endpoint trả JSON response với status, duration_seconds, và chi tiết kết quả

---

## 9. Verification Guide

### 9.1 News Articles

```sql
-- Đếm bài theo source (24 giờ gần nhất)
SELECT source, COUNT(*) FROM articles
WHERE published_at > NOW() - INTERVAL '24 hours'
GROUP BY source;

-- Kiểm tra freshness
SELECT source, MAX(published_at) as latest, NOW() - MAX(published_at) as age
FROM articles GROUP BY source;

-- Kiểm tra embedding status
SELECT source, embedded, COUNT(*) FROM articles
GROUP BY source, embedded ORDER BY source;
```

**Expected patterns:**
- Mỗi source (vietstock, cafef, vneconomy) nên có bài mới mỗi 6-7 giờ
- `published_at` age không nên > 24 giờ cho ngày bình thường
- `embedded=TRUE` ratio nên cao (>90% cho bài >1 giờ tuổi)

### 9.2 Macro News (Category-Based)

```sql
-- Đếm tin vĩ mô theo source (24 giờ gần nhất)
SELECT source, COUNT(*) FROM articles
WHERE category = 'macro' AND published_at > NOW() - INTERVAL '24 hours'
GROUP BY source;

-- Kiểm tra tỷ lệ phân loại
SELECT category, COUNT(*) FROM articles
WHERE published_at > NOW() - INTERVAL '7 days'
GROUP BY category;
```

**Expected patterns:**
- Mỗi source nên có tin macro mới mỗi 6-7 giờ
- Tỷ lệ macro/stock phụ thuộc vào feed config (~50/50)

### 9.3 Stock Prices

```sql
-- Row counts theo ticker
SELECT ticker_symbol, COUNT(*), MAX(data_as_of) as latest
FROM market_data
WHERE data_type = 'stock_price'
GROUP BY ticker_symbol
ORDER BY ticker_symbol;

-- Kiểm tra OHLCV completeness cho 1 ticker
SELECT ticker_symbol, data_as_of,
       open_price, high_price, low_price, close_price, volume
FROM market_data
WHERE data_type = 'stock_price' AND ticker_symbol = 'VNM'
ORDER BY data_as_of DESC LIMIT 5;

-- Tickers thiếu dữ liệu gần đây
SELECT ticker_symbol, MAX(data_as_of) as latest,
       NOW() - MAX(data_as_of) as age
FROM market_data
WHERE data_type = 'stock_price'
GROUP BY ticker_symbol
HAVING NOW() - MAX(data_as_of) > INTERVAL '3 days'
ORDER BY age DESC;
```

**Expected patterns:**
- Mỗi ticker nên có ≥30 rows sau initial load (1 năm data)
- `data_as_of` latest nên là ngày giao dịch gần nhất
- OHLCV: tất cả 5 columns phải NOT NULL cho stock_price records

### 9.4 Technical Indicators

```sql
-- Đếm indicators theo tên
SELECT indicator_name, COUNT(DISTINCT ticker_symbol) as tickers,
       MAX(data_as_of) as latest
FROM market_data
WHERE data_type = 'technical_indicator'
GROUP BY indicator_name
ORDER BY indicator_name;

-- Kiểm tra indicators cho 1 ticker cụ thể
SELECT indicator_name, indicator_value, data_as_of
FROM market_data
WHERE data_type = 'technical_indicator'
  AND ticker_symbol = 'VNM'
  AND data_as_of = (
    SELECT MAX(data_as_of) FROM market_data
    WHERE data_type = 'technical_indicator' AND ticker_symbol = 'VNM'
  )
ORDER BY indicator_name;
```

**Expected patterns:**
- ~44 tickers × lên đến 18 indicator values (MACD có 3, Volume có 3, BB có 3, DC có 3)
- SMA_200 có thể thiếu cho tickers mới (cần ≥200 phiên data)
- RS_VNINDEX cần data VN-Index — nếu thiếu sẽ bị skip

### 9.5 Embeddings

```sql
-- Tỷ lệ embedded
SELECT embedded, COUNT(*) as count,
       ROUND(COUNT(*)::numeric / SUM(COUNT(*)) OVER() * 100, 1) as percentage
FROM articles GROUP BY embedded;

-- Kiểm tra vector dimension
SELECT id, title, array_length(embedding::real[], 1) as dim
FROM articles
WHERE embedded = TRUE
LIMIT 5;

-- Bài chưa embedded (cần xử lý)
SELECT id, title, source, published_at, created_at
FROM articles
WHERE embedded = FALSE
ORDER BY created_at DESC LIMIT 10;
```

**Expected patterns:**
- Vector dimension phải = 1536
- Bài mới có thể chưa embedded (chờ pipeline chạy)
- Embedded ratio nên >95% cho bài >1 giờ tuổi

---

## 10. Troubleshooting Guide

### 10.1 Crawl Failures

**Triệu chứng:** Không có bài mới trong `articles` table

**Nguyên nhân & Cách xử lý:**

| Vấn Đề | Log Pattern | Cách Xử Lý |
|---------|-------------|-------------|
| Network error | `httpx.ConnectError` | Kiểm tra network connectivity, DNS resolution |
| Rate limit exceeded | HTTP 429, `httpx.HTTPStatusError` | Giảm `rate_limit_rps` trong `sources.yaml`, chờ 1 giờ |
| robots.txt blocked | `URL disallowed by robots.txt` | Kiểm tra `https://{domain}/robots.txt`, cập nhật User-Agent nếu cần |
| RSS feed thay đổi | Parse error, empty items | Kiểm tra RSS feed URL còn hoạt động, cập nhật URL trong `sources.yaml` |
| HTML structure thay đổi | Article body rỗng | Cập nhật CSS selectors trong crawler tương ứng |
| Timeout | `httpx.TimeoutException` | Tăng timeout (hiện tại 30s) hoặc kiểm tra target site |

### 10.2 Missing Data

**Triệu chứng:** Data không cập nhật đúng lịch

| Vấn Đề | Kiểm Tra | Cách Xử Lý |
|---------|----------|-------------|
| Không có bài mới | `SELECT MAX(published_at) FROM articles WHERE source='...'` | Kiểm tra RSS feeds, crawl logs |
| Skipped trading day | Log: `Non-trading day, skipping incremental` | Bình thường — chỉ crawl vào ngày giao dịch |
| Ticker bị skip | Log: `Invalid ticker format` | Kiểm tra `stock_tickers.yaml`, pattern `^[A-Z]{3,4}$` |
| Mock data thay vì real | `data_source='mock'` trong market_data | vnstock API fail — kiểm tra connectivity, API status |

### 10.3 Embedding Failures

**Triệu chứng:** `embedded=FALSE` tỷ lệ cao

| Vấn Đề | Kiểm Tra | Cách Xử Lý |
|---------|----------|-------------|
| OpenAI API error | Log: `RateLimitError`, `APITimeoutError` | Kiểm tra API key, quota, billing |
| Batch size quá lớn | Log: `APIConnectionError` | Giảm batch_size (default 100) |
| Không có content | Log: `Skipping article — no content` | Bài chưa có raw_content (crawl fail trước đó) |
| API key missing | `OPENAI_API_KEY` not set | Set environment variable trong Docker/`.env` |

### 10.4 Scheduler Issues

**Triệu chứng:** Pipeline không chạy đúng lịch

| Vấn Đề | Kiểm Tra | Cách Xử Lý |
|---------|----------|-------------|
| Prefect flow stuck | Prefect UI → Flow Runs → check state | Cancel flow run, investigate logs |
| Missed schedule | Prefect UI → Deployments → check schedule | Verify cron expression, timezone, container uptime |
| Container restart | `docker logs scheduler` | Kiểm tra OOM, disk space, health checks |
| App unreachable | Log: `ConnectError` từ scheduler | Kiểm tra app container running, network `http://app:8000` |
| Auth failed | HTTP 403 từ internal endpoints | Verify header `X-Trigger-Source: prefect-scheduler` |

Xem danh sách deployments đang serve
`prefect deployment ls`

Xem chi tiết 1 flow run
`prefect flow-run inspect <flow-run-id>`

### 10.5 Database Issues

**Triệu chứng:** Lỗi khi read/write data

| Vấn Đề | Kiểm Tra | Cách Xử Lý |
|---------|----------|-------------|
| Connection pool exhausted | Log: `TimeoutError` on DB connect | Tăng pool size, kiểm tra connection leaks |
| pgvector index issues | Slow semantic search queries | `REINDEX INDEX idx_articles_embedding_hnsw;` |
| Disk space | `df -h` trên DB server | Clean old data, tăng disk, chạy lifecycle pipeline |
| Migration mismatch | Alembic version mismatch | `alembic upgrade head` |
| Duplicate key | `IntegrityError` on insert | Bình thường cho articles (dedup by URL) |
