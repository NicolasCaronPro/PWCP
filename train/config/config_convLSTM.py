import configparser
from pathlib import Path
import torch

def check_and_create_path(path: Path):
    """
    Creer un dossier s'il n'existe pas
    """
    path_way = path.parent if path.is_file() else path

    path_way.mkdir(parents=True, exist_ok=True)

    if not path.exists():
        path.touch()

### Global param
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seq_len = 31
batch_size = 2

### Path and Variable
dataset_version = '1.0'
model_version = '6.0'
root = Path('/Home/Users/ncaron/Bureau/Dual_Model_Research/Model/LargeScale')

dir_csv = root / dataset_version
dir_output = root / dataset_version / "models" / "convLSTM" / model_version

check_and_create_path(Path(dir_csv  / "models" / "convLSTM"))
check_and_create_path(Path(dir_csv  / "models" / "convLSTM" / model_version))
check_and_create_path(Path(dir_csv  / "models" / "convLSTM" / model_version / 'METRICS'))
check_and_create_path(Path(dir_csv  / "models" / "convLSTM" / model_version / "model"))

topvariables = [ 'dc_0', 'ffmc_0', 'dmc_0', 'isi_0', 'bui_0', 'fwi_0',
 'nesterov_0', 'munger_0', 'kbdi_0', 'angstroem_0', 'daily_severity_rating_0',
 'temp_0', 'dwpt_0', 'rhum_0', 'prcp_0', 'wdir_0', 'wspd_0', 'prec24h_0',
 'temp16_0', 'dwpt16_0', 'rhum16_0', 'prcp16_0', 'wdir16_0', 'wspd16_0', 'prec24h16_0',
 'days_since_rain_0',
 'dc_mean', 'dmc_mean', 'ffmc_mean', 'isi_mean', 'bui_mean', 'fwi_mean', 'daily_severity_rating_mean', 'temp_mean', 'dwpt_mean', 'rhum_mean', 'prcp_mean', 'prec24h_mean', 'wspd_mean',
 'dc_max','dmc_max', 'ffmc_max', 'isi_max', 'bui_max', 'fwi_max', 'daily_severity_rating_max', 'temp_max', 'dwpt_max', 'rhum_max', 'prcp_max', 'prec24h_max', 'wspd_max',
 'dc_min', 'dmc_min', 'ffmc_min', 'isi_min', 'bui_min', 'fwi_min', 'daily_severity_rating_min',
 'temp_min', 'dwpt_min', 'rhum_min', 'prcp_min', 'prec24h_min', 'wspd_min',
 'temp_gradient','dwpt_gradient', 'rhum_gradient', 'prcp_gradient', 'prec24h_gradient', 'wspd_gradient',
 'temp16_mean', 'dwpt16_mean', 'rhum16_mean', 'prcp16_mean', 'prec24h16_mean', 'wspd16_mean',
 'temp16_max', 'dwpt16_max', 'rhum16_max', 'prcp16_max', 'prec24h16_max', 'wspd16_max',
 'temp16_min', 'dwpt16_min', 'rhum16_min', 'prcp16_min', 'prec24h16_min', 'wspd16_min',
 'temp_16gradient', 'dwpt_16gradient', 'rhum_16gradient', 'prcp_16gradient', 'prec24h_16gradient', 'wspd_16gradient'
]

variables = [  'dc_0', 'ffmc_0', 'dmc_0', 'isi_0', 'bui_0', 'fwi_0',
 'nesterov_0', 'munger_0', 'kbdi_0', 'angstroem_0', 'daily_severity_rating_0',
 'temp_0', 'dwpt_0', 'rhum_0', 'prcp_0', 'wdir_0', 'wspd_0', 'prec24h_0',
 'temp16_0', 'dwpt16_0', 'rhum16_0', 'prcp16_0', 'wdir16_0', 'wspd16_0', 'prec24h16_0',
 'days_since_rain_0',

'dc_mean', 'dmc_mean', 'ffmc_mean', 'isi_mean', 'bui_mean', 'fwi_mean', 'daily_severity_rating_mean', 'temp_mean', 'dwpt_mean', 'rhum_mean', 'prcp_mean', 'prec24h_mean', 'wspd_mean',
 'dc_max','dmc_max', 'ffmc_max', 'isi_max', 'bui_max', 'fwi_max', 'daily_severity_rating_max', 'temp_max', 'dwpt_max', 'rhum_max', 'prcp_max', 'prec24h_max', 'wspd_max',
 'dc_min', 'dmc_min', 'ffmc_min', 'isi_min', 'bui_min', 'fwi_min', 'daily_severity_rating_min',
 'temp_min', 'dwpt_min', 'rhum_min', 'prcp_min', 'prec24h_min', 'wspd_min',
 'temp_gradient','dwpt_gradient', 'rhum_gradient', 'prcp_gradient', 'prec24h_gradient', 'wspd_gradient',
 'temp16_mean', 'dwpt16_mean', 'rhum16_mean', 'prcp16_mean', 'prec24h16_mean', 'wspd16_mean',
 'temp16_max', 'dwpt16_max', 'rhum16_max', 'prcp16_max', 'prec24h16_max', 'wspd16_max',
 'temp16_min', 'dwpt16_min', 'rhum16_min', 'prcp16_min', 'prec24h16_min', 'wspd16_min',
 'temp_16gradient', 'dwpt_16gradient', 'rhum_16gradient', 'prcp_16gradient', 'prec24h_16gradient', 'wspd_16gradient',

 'month', 'dayofyear', 'dayofweek',

 'population_mean', 'elevation_mean', 'elevation_max', 'elevation_min',
 '0.0','1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0',
 #'NDVI', 'NDMI', 'NDSI', 'NDBI', 'NDWI',
 'NumberOfCities',
 'bankHolidays', 'bankHolidaysEve', 'bankHolidaysEveEve',
 'holidays', 'holidaysEve', 'holidaysEveEve', 'holidaysLastDay', 'holidaysLastLastDay',
 'daylightSavingTime',
 'confinement1', 'confinement2', 'ramadan',
 # 'sunRised', 'moonphase','moonrised', 'moon_distance', 'sun_distance',
 'grippe_inc', 'diarrhee_inc', 'varicelle_inc', 'ira_inc',
 'Radio_flux_10cm', 'SESC_Sunspot_number', 'Sunspot_area',
 'New_regions', 'XrayC', 'XrayM', 'XrayX', 'XrayS',
 'Optical1', 'Optical2', 'Optical3',
 'Canicule',
 'match_LGF1', 'match_CL',
 'match_LGF1-2', 'match_LGF1-4', 'match_LGF1-6',  'match_LGF1-8', 'match_LGF1-10', 'match_LGF1-12',
 'match_CL-2', 'match_CL-4', 'match_CL-6', 'match_CL-8', 'match_CL-10', 'match_CL-12',
 'PM25', 'O3A', 'O3B', 'PM10A', 'PM10B', 'NO2A', 'NO2B',
 ]

for top in ['_top1', '_top2', '_top3']:
    for var in topvariables:
        variables.append(var+top)