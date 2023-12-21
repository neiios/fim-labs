import yfinance as yf

symbol = "META"
start_date = "2022-01-01"
end_date = "2023-09-01"

start_date2 = "2023-09-01"
end_date2 = "2023-12-01"

start_date3 = "2021-11-01"
end_date3 = "2023-10-31"

ticker = yf.Ticker(symbol)

hist = ticker.history(interval="1d", start=start_date, end=end_date)
hist.to_csv(f"{symbol}_{start_date}_{end_date}.csv")

hist = ticker.history(interval="1d", start=start_date2, end=end_date2)
hist.to_csv(f"{symbol}_{start_date2}_{end_date2}.csv")

hist = ticker.history(interval="1d", start=start_date3, end=end_date3)
hist.to_csv(f"{symbol}_{start_date3}_{end_date3}.csv")
