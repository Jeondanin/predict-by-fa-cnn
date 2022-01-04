import numpy as np
from talib import *

close = np.random.random(100)

output = SMA(close)
print(output)

# 9. EMA - Exponential Moving Average
real = EMA(close, timeperiod=30)

# 10. Double Exponential Moving Average
dema = DEMA(close, timeperiod=30)

# 11. Double Exponential Moving Average
real = T3(close, timeperiod=5, vfactor=0)

# 12, 13, 14 MACD - Moving Average Convergence/Divergence
macd, macdsignal, macdhist = MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)

# 15. money_flow 
money_flow = MFI(high, low, close, volume, timeperiod=14)

# 16. momentum
momentum = MOM(close, timeperiod=10)

# 17. ADX - Average Directional Movement Index
adx = ADX(high, low, close, timeperiod=14)

# 18. ROC - Rate of change : ((price/prevPrice)-1)*100
roc = ROC(close, timeperiod=10)

# 19. RSI - Relative Strength Index
rsi = RSI(close, timeperiod=14)

# 20. Commodity Channel Index
cci = CCI(high, low, close, timeperiod=14)

# 21. 1-day ROC of a triple smooth EMA
trix = TRIX(close, timeperiod=30)

# 22. Williams’ %R
william = WILLR(high, low, close, timeperiod=14)

# 23.  Balance of Power
bop = BOP(open, high, low, close)

# 24. Absolute Price Oscillator
apo = APO(close, fastperiod=12, slowperiod=26, matype=0)

# 25. Average Directional Movement Index Rating (ADXR) 
adxr = ADXR(high, low, close, timeperiod=14)

# 26. Average Directional Movement Index (ADX) 
adx = ADX(high, low, close, timeperiod=14)

# 27.  Ultimate Oscillator
ultsoc = ULTOSC(high, low, close, timeperiod1=7, timeperiod2=14, timeperiod3=28)

# 28. AD - Chaikin A/D Line
ad = AD(high, low, close, volume)

# 29. ADOSC - Chaikin A/D Oscillator
adosc = ADOSC(high, low, close, volume, fastperiod=3, slowperiod=10)

# 30. real = OBV(close, volume)
obv = OBV(close, volume)

# 31. ATR - Average True Range
real = ATR(high, low, close, timeperiod=14)

# 32. TRANGE - True Range
trange = TRANGE(high, low, close)



