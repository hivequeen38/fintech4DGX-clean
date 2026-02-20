from datetime import datetime
import get_daily_results
import get_historical_html
import mainDeltafromToday
import NVDA_param
import PLTR_param
import APP_param
import META_param
import MSTR_param
import INOD_param
import QQQ_param
import SMCI_param
import TSM_param
import ANET_param
import CRDO_param
import ALAB_param
import TSLA_param

# ['NVDA', 'PLTR', 'APP', 'ANET', 'CRDO', 'ALAB' ]
today_date_str = datetime.now().strftime("%Y-%m-%d")
today_date_str = '2026-02-19'

response = input("Upload results to Google Cloud? (Y/N): ").strip().upper()
upload_to_cloud = response == 'Y'
print(f"Cloud upload: {'ON' if upload_to_cloud else 'OFF - results will be saved locally only'}")

mainDeltafromToday.main(CRDO_param.reference, end_date = today_date_str)
mainDeltafromToday.main(NVDA_param.reference, end_date = today_date_str)
get_historical_html.upload_all_results(today_date_str, upload_to_cloud=upload_to_cloud)

mainDeltafromToday.main(PLTR_param.reference, end_date = today_date_str)
mainDeltafromToday.main(APP_param.reference, end_date = today_date_str)
mainDeltafromToday.main(INOD_param.reference, end_date = today_date_str)

mainDeltafromToday.main(CRDO_param.AAII_option_vol_ratio, end_date = today_date_str)
mainDeltafromToday.main(NVDA_param.AAII_option_vol_ratio, end_date = today_date_str)
get_historical_html.upload_all_results(today_date_str, upload_to_cloud=upload_to_cloud)

mainDeltafromToday.main(PLTR_param.AAII_option_vol_ratio, end_date = today_date_str)
mainDeltafromToday.main(APP_param.AAII_option_vol_ratio, end_date = today_date_str)
mainDeltafromToday.main(INOD_param.AAII_option_vol_ratio, end_date = today_date_str)

get_historical_html.upload_all_results(today_date_str, upload_to_cloud=upload_to_cloud)
