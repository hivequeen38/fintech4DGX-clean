import processPrediction

date_str = '2024-04-05'
symbol = 'NVDA'
close = 880.08
result = ['DOWN', 'UP', 'No Change', 'Down', 'No Change']

processPrediction.process_prediction_results(symbol, date_str, close, result)

date_str = '2024-04-06'
symbol = 'NVDA'
close = 880.09
result = ['DOWN', 'DOWN', 'No Change', 'Down', 'No Change']

processPrediction.process_prediction_results(symbol, date_str, close, result)

date_str = '2024-04-05'
symbol = 'NVDA'
close = 900
result = ['UP', 'UP', 'No Change', 'Down', 'No Change']

processPrediction.process_prediction_results(symbol, date_str, close, result)
