import pandas as pd


def ichimoku(df):
    hi, lo = df["high"], df["low"]
    tenkan = (hi.rolling(9).max() + lo.rolling(9).min()) / 2
    kijun = (hi.rolling(26).max() + lo.rolling(26).min()) / 2
    span_a = ((tenkan + kijun) / 2).shift(26)
    span_b = ((hi.rolling(52).max() + lo.rolling(52).min()) / 2).shift(26)
    return tenkan, kijun, span_a, span_b


def get_signal(df):
    """df: کندل‌های بسته‌شده. خروجی: (LONG/SHORT, زمان کندل) یا None"""
    if len(df) < 90:
        return None
    t, k, a, b = ichimoku(df)
    top = pd.concat([a, b], axis=1).max(axis=1)
    bot = pd.concat([a, b], axis=1).min(axis=1)
    close = df["close"]

    # کندل آخر و یکی قبلش (برای تحمل تأخیر GitHub Actions)
    for i in (-1, -2):
        if pd.isna(top.iloc[i]) or pd.isna(k.iloc[i - 1]):
            continue
        cross_up = t.iloc[i - 1] <= k.iloc[i - 1] and t.iloc[i] > k.iloc[i]
        cross_dn = t.iloc[i - 1] >= k.iloc[i - 1] and t.iloc[i] < k.iloc[i]
        ts = int(df["t"].iloc[i])
        if cross_up and close.iloc[i] > top.iloc[i]:
            return "LONG", ts
        if cross_dn and close.iloc[i] < bot.iloc[i]:
            return "SHORT", ts
    return None
