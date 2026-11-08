# Data Readme

## `data/hf_regime_data.db` and `data/trading_data.db`

SQLite3 databases containing ~5 years of 1-minute OHLCV candle data downloaded from [Polygon via Massive](https://massive.ai/). Both files share the identical schema (described once below).

- `data/hf_regime_data.db` (~2.1 GB): candle data for a basket of 20 ETFs used to capture high-frequency market regime signals — `ARKK`, `DIA`, `GLD`, `HYG`, `IWM`, `KBE`, `QQQ`, `SMH`, `SPY`, `TLT`, `USD`, `XBI`, `XLE`, `XLF`, `XLI`, `XLK`, `XLP`, `XLU`, `XLV`, `XLY`. 12,873,953 rows in `bars`, date range `2020-09-08` to `2026-09-07`.
- `data/trading_data.db` (~2.8 GB): candle data for 23 selected individual stock tickers — `AAL`, `AAPL`, `AMD`, `AMZN`, `BAC`, `BMNR`, `F`, `GOOGL`, `INTC`, `IREN`, `META`, `MSFT`, `MU`, `NOK`, `NU`, `NVDA`, `ONDS`, `PATH`, `PCG`, `PLTR`, `PLUG`, `SPCX`, `TSLA`. 17,017,959 rows in `bars`, date range `2021-09-08` to `2026-09-07`.

Both databases also carry `.db-shm`/`.db-wal` sidecar files (WAL journal mode).

### Table: `bars`

One row per 1-minute OHLCV candle per ticker.

| Column | Type | Description |
| --- | --- | --- |
| `id` | INTEGER | Autoincrement primary key |
| `ticker` | TEXT | Ticker symbol |
| `timestamp` | BIGINT | Bar start time, Unix epoch milliseconds (UTC) |
| `open` | REAL | Opening price |
| `high` | REAL | High price |
| `low` | REAL | Low price |
| `close` | REAL | Closing price |
| `volume` | BIGINT | Traded volume |
| `vwap` | REAL | Volume-weighted average price for the bar |
| `transactions` | INTEGER | Number of trades in the bar |
| `created_at` | TIMESTAMP | Row insertion time (default `CURRENT_TIMESTAMP`) |

Unique constraint on `(ticker, timestamp)`. Indexes: `idx_bars_ticker`, `idx_bars_timestamp`, `idx_bars_ticker_timestamp`.

### Table: `dividends`

Dividend events per ticker.

| Column | Type | Description |
| --- | --- | --- |
| `id` | INTEGER | Autoincrement primary key |
| `ticker` | TEXT | Ticker symbol |
| `ex_dividend_date` | TEXT | Ex-dividend date (`YYYY-MM-DD`) |
| `cash_amount` | REAL | Dividend cash amount per share |
| `declaration_date` | TEXT | Date dividend was declared |
| `dividend_type` | TEXT | Dividend type (e.g. regular cash) |
| `frequency` | INTEGER | Payout frequency code |
| `pay_date` | TEXT | Payment date |
| `record_date` | TEXT | Record date |
| `created_at` | TIMESTAMP | Row insertion time |

Unique constraint on `(ticker, ex_dividend_date)`. Indexes: `idx_dividends_ticker`, `idx_dividends_ex_dividend_date`.

### Table: `splits`

Stock split events per ticker.

| Column | Type | Description |
| --- | --- | --- |
| `id` | INTEGER | Autoincrement primary key |
| `ticker` | TEXT | Ticker symbol |
| `execution_date` | TEXT | Split execution date (`YYYY-MM-DD`) |
| `split_from` | REAL | Pre-split share count basis |
| `split_to` | REAL | Post-split share count basis |
| `created_at` | TIMESTAMP | Row insertion time |

Unique constraint on `(ticker, execution_date)`. Indexes: `idx_splits_ticker`, `idx_splits_execution_date`.

### Table: `options`

Options contract reference data per underlying ticker (currently empty — 0 rows — in both databases).

| Column | Type | Description |
| --- | --- | --- |
| `id` | INTEGER | Autoincrement primary key |
| `ticker` | TEXT | Option contract ticker (unique) |
| `contract_type` | TEXT | `call` or `put` |
| `exercise_style` | TEXT | e.g. `american`, `european` |
| `expiration_date` | TEXT | Contract expiration date |
| `primary_exchange` | TEXT | Primary listing exchange |
| `shares_per_contract` | INTEGER | Shares per contract |
| `strike_price` | REAL | Strike price |
| `underlying_ticker` | TEXT | Underlying equity/ETF ticker |
| `cfi` | TEXT | CFI classification code |
| `additional_underlyings` | TEXT | Additional underlying tickers, if any |
| `correction` | INTEGER | Correction flag/counter |
| `created_at` | TIMESTAMP | Row insertion time |

Indexes: `idx_options_underlying_ticker`, `idx_options_expiration_date`, `idx_options_strike_price`.

### Table: `metadata`

Bookkeeping table tracking ingestion status per ticker/data type.

| Column | Type | Description |
| --- | --- | --- |
| `id` | INTEGER | Autoincrement primary key |
| `ticker` | TEXT | Ticker symbol |
| `data_type` | TEXT | Data category (`bars`, `dividends`, `splits`) |
| `last_update` | TIMESTAMP | Last time this ticker/data_type was refreshed |
| `records_count` | INTEGER | Number of records ingested for this ticker/data_type |
| `start_date` | TEXT | Earliest date covered |
| `end_date` | TEXT | Latest date covered |

Unique constraint on `(ticker, data_type)`. Indexes: `idx_metadata_ticker`, `idx_metadata_data_type`.

## `data/news_6year.db`

SQLite3 database containing ~6 years of financial news data downloaded from [Benzinga](https://www.benzinga.com/), along with precomputed sentence embeddings for each article. Size on disk: ~4.9 GB.

- Row count: 1,459,119 articles
- Date range (`created`): `2020-09-27T00:07:48Z` to `2026-09-25T23:50:00Z`

### Table: `news_embeddings`

One row per news article.

| Column | Type | Description |
| --- | --- | --- |
| `id` | REAL | Benzinga article ID (stored as float) |
| `created` | TEXT | Article creation timestamp (ISO 8601, UTC) |
| `updated` | TEXT | Article last-updated timestamp (ISO 8601, UTC) |
| `title` | TEXT | Article headline |
| `teaser` | TEXT | Short summary/body excerpt (HTML-entity encoded) |
| `channels` | TEXT | Comma-separated Benzinga channel tags (e.g. `earnings`, `after-hours center`) |
| `categories` | TEXT | Comma-separated category tags |
| `tags` | TEXT | Comma-separated free-form tags |
| `author` | TEXT | Article author/source (e.g. `benzinga insights`) |
| `url` | TEXT | Link to the original Benzinga article |
| `ticker_symbols` | TEXT | Comma-separated stock ticker symbols referenced in the article |
| `ticker_names` | REAL | Ticker names (column is typed REAL but appears unused/empty in practice) |
| `combined_embedding` | TEXT | Serialized sentence-transformer embedding vector for the combined article text |

Indexes: `idx_created` on `created`, `idx_id` on `id`.

### Table: `encoding_metadata`

Single-row table describing how the embeddings were generated.

| Column | Type | Description |
| --- | --- | --- |
| `model_type` | TEXT | Embedding model used (`sentence-transformer`) |
| `embedding_dim` | INTEGER | Embedding vector dimensionality (`768`) |
| `combine_fields` | INTEGER | Boolean flag (1) indicating multiple text fields were combined before encoding |
| `use_float16` | INTEGER | Boolean flag (1) indicating embeddings are stored as float16 |
| `total_articles` | INTEGER | Total articles encoded at time of generation (`1,459,126`) |
| `encoded_at` | TEXT | Timestamp when encoding was run |

## `data/fred_econ_data_6yrs.csv`

CSV file containing ~6 years of U.S. macroeconomic indicator data downloaded from [FRED](https://fred.stlouisfed.org/) (Federal Reserve Economic Data). Size on disk: ~48 KB.

- Row count: 1,586 rows (one row per calendar date; 1,587 lines including header)
- Date range: `2020-07-01` to `2026-09-24`
- Index: first (unnamed) column is the observation date (`YYYY-MM-DD`)

Each FRED series is reported at its own native frequency (daily, monthly, or quarterly) and has been merged onto a common daily date index, so most rows contain values for only a subset of columns; the remaining columns are blank/NaN until their next reporting date.

| Column | FRED Series ID | Frequency | Description |
| --- | --- | --- | --- |
| `FEDFUNDS` | FEDFUNDS | Monthly | Effective Federal Funds Rate (%) |
| `CPIAUCSL` | CPIAUCSL | Monthly | Consumer Price Index for All Urban Consumers: All Items (seasonally adjusted) |
| `PPIACO` | PPIACO | Monthly | Producer Price Index: All Commodities |
| `GDPC1` | GDPC1 | Quarterly | Real Gross Domestic Product (chained dollars, seasonally adjusted annual rate) |
| `UNRATE` | UNRATE | Monthly | Civilian Unemployment Rate (%) |
| `PAYEMS` | PAYEMS | Monthly | Total Nonfarm Payroll Employment (thousands of persons) |
| `CPILFESL` | CPILFESL | Monthly | Core CPI: All Items Less Food and Energy (seasonally adjusted) |
| `DGS10` | DGS10 | Daily | 10-Year Treasury Constant Maturity Rate (%) |
| `DGS2` | DGS2 | Daily | 2-Year Treasury Constant Maturity Rate (%) |
| `PCE` | PCE | Monthly | Personal Consumption Expenditures (billions of dollars) |

**Notes:**
- Daily series (`DGS10`, `DGS2`) are populated only on business days; weekends/holidays are blank.
- Monthly/quarterly series are populated only on their release date row and blank elsewhere; forward-filling or resampling is typically needed before use in time-series models.
