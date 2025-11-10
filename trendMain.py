import logging
import trendConfig
from datetime import datetime
import trendAnalysis
import analysisUtil
import multiZoneAnalysisNew
import NVDA_param
import PLTR_param
import APP_param
import META_param
import MSTR_param
import INOD_param
import QQQ_param
import CRDO_param
import TSM_param

print("All libraries loaded for "+ __file__)



#######
# MAIN LOOP enters here

logging.basicConfig(level=logging.INFO)

# elements = [0.007, 0.01, 0.013]
# for item in elements:
#     param["threshold"] = 0.01

param = PLTR_param.AAII_option_vol_ratio

# today_date_str = datetime.now().strftime("%Y-%m-%d")
today_date_str = '2025-05-08'
param["comment"]='debug AAIIoption ratio for PLTR with days to reprt'
param["end_date"] = today_date_str
# param['target_size'] = 1
# trendAnalysis.load_data_to_cache(trendConfig.config, param, use_global_cache=True)
trendAnalysis.analyze_trend(trendConfig.config, param, False, use_regime=False, regime_choice='LOW_VOL', use_globl_cache=False)
# multiZoneAnalysisNew.analyze_trend(trendConfig.config, param, "debug MZ zone more dim and layers", True, True)
# trendAnalysis.analyze_trend_timesplit(trendConfig.config, param, True)
