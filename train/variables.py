meteoVariable = [ 'dc', 
                 'ffmc',
                 'dmc',
                 'isi',
                 'bui',
                 'fwi',
 'nesterov', 'munger', 'kbdi', 'angstroem', 'daily_severity_rating',
 'temp', 'dwpt', 'rhum', 'prcp', 'wdir', 'wspd', 'prec24h',
 'temp16', 'dwpt16', 'rhum16', 'prcp16', 'wdir16', 'wspd16', 'prec24h16',
 'days_since_rain'
]

topoVariables = [ 'NDVI', 'NDMI', 'NDSI', 'NDWI',
 'population',
 'elevation',
 'highway'
 ]

topvariables = []

variables = [
 #'day_since_fire',
 'month',
 'dayofyear', 'dayofweek',

 'bankHolidays', 'bankHolidaysEve', 'bankHolidaysEveEve',
 'holidays', 'holidaysEve', 'holidaysEveEve', 'holidaysLastDay', 'holidaysLastLastDay',
 '0.0','1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0',
 'influenceNATURELSFirebefore'
 #'influenceInondationbefore'
 #'daylightSavingTime',
 #'confinement1', 'confinement2', 'ramadan',
 # 'sunRised', 'moonphase','moonrised', 'moon_distance', 'sun_distance',
 #'grippe_inc', 'diarrhee_inc', 'varicelle_inc', 'ira_inc',
 #'Radio_flux_10cm', 'SESC_Sunspot_number', 'Sunspot_area',
 #'New_regions', 'XrayC', 'XrayM', 'XrayX', 'XrayS',
 #'Optical1', 'Optical2', 'Optical3',
 #'Canicule',
 #'match_LGF1', 'match_CL',
 #'match_LGF1-2', 'match_LGF1-4', 'match_LGF1-6',  'match_LGF1-8', 'match_LGF1-10', 'match_LGF1-12',
 #'match_CL-2', 'match_CL-4', 'match_CL-6', 'match_CL-8', 'match_CL-10', 'match_CL-12',
 #'PM25', 'O3A', 'O3B', 'PM10A', 'PM10B', 'NO2A', 'NO2B',
 ]

for i in range(0, 1):
    for var in meteoVariable:
        if var == 'days_since_rain' and i > 0:
            continue
        for method in ['mean', 'max', 'min', 'std']:
            variables.append(var+'_'+method+'_'+str(i))
            #topvariables.append(var+'_'+method+'_'+str(i))

for var in topoVariables:
    for method in ['mean', 'max', 'min', 'std']:
        variables.append(var+'_'+method)
        topvariables.append(var+'_'+method)