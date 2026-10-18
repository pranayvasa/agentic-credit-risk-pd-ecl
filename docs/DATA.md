# Data: Home Credit Default Risk (public, Kaggle)

This project uses only the public [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk/data) competition data. No raw data is committed: `data/` is gitignored.

## Download

1. Create a Kaggle account, open the competition page and accept the competition rules (Kaggle requires this before download).
2. Either download the zip from the **Data** tab in your browser, or use the Kaggle CLI with your own API token (`~/.kaggle/kaggle.json`):

   ```bash
   pip install kaggle
   kaggle competitions download -c home-credit-default-risk -p data/raw
   unzip data/raw/home-credit-default-risk.zip -d data/raw
   ```

3. Check that these files are in `data/raw/`:

   | File | Rows | Leg |
   |---|---|---|
   | `application_train.csv` | 307,511 | Application (and TARGET) |
   | `bureau.csv` | 1,716,428 | Bureau |
   | `bureau_balance.csv` | 27,299,925 | Bureau |
   | `installments_payments.csv` | 13,605,401 | Behavioural |
   | `POS_CASH_balance.csv` | 10,001,358 | Behavioural |
   | `credit_card_balance.csv` | 3,840,312 | Behavioural |
   | `previous_application.csv` | 1,670,214 | Previous applications |

`application_test.csv` has no TARGET and is not used.

## Cache

The first run converts the CSVs into a downcast parquet cache in `data/interim/` (about 20 seconds; 2.5 GB of CSV becomes about 513 MB). Later runs reuse it, and the cache rebuilds automatically if a source file changes.
