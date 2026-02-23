import mainDeltafromToday
import NVDA_param
import get_historical_html

today_date_str = '2026-02-20'
upload_to_cloud = True

print(f"Running NVDA only — end_date={today_date_str}, cloud upload={upload_to_cloud}")

mainDeltafromToday.main(NVDA_param.reference, end_date=today_date_str)
mainDeltafromToday.main(NVDA_param.AAII_option_vol_ratio, end_date=today_date_str,
                        input_comment='(AAII_option_vol_ratio) dte added')

get_historical_html.upload_all_results(today_date_str, upload_to_cloud=upload_to_cloud)

print("NVDA training complete.")
