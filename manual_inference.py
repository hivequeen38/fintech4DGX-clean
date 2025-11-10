from datetime import datetime
import daily_main

today_date_str = datetime.now().strftime("%Y-%m-%d")
# today_date_str = '2024-10-28'
daily_main.NVDA_inference(today_date_str)
daily_main.PLTR_inference(today_date_str)