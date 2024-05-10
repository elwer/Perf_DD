import arff
from scipy.io import arff
import pandas as pd
import numpy as np
import pickle

base_path="../../data/"

def getAirlines():
    data = arff.loadarff(base_path + 'airlines.arff')
    # Convert the ARFF data to a DataFrame
    df = pd.DataFrame(data[0])

    df['target'] = df['class'].apply(lambda x: int(x))

    col_list = ['Flight', 'Time', 'Length', 'Airline', 'AirportFrom','AirportTo', 'DayOfWeek', 'target']
    df = df[col_list]

    df['DayOfWeek'] = df['DayOfWeek'].apply(lambda x: int(x.decode()))

    airlineSet = {b'9E', b'AA', b'AS', b'B6', b'CO', b'DL', b'EV', b'F9', b'FL', b'HA', b'MQ', b'OH', b'OO', b'UA', b'US', b'WN', b'XE', b'YV'}
    airlineList = []
    for s in airlineSet:
        airlineList.append(s)

    # replace_func = lambda x: 'bad.' if x in values_to_replace else x
    df['Airline'] = df['Airline'].apply(lambda x: airlineList.index(x))

    airportFromSet = {b'ABE',b'ABI',b'ABQ',b'ABR',b'ABY',b'ACT',b'ACV',b'ACY',b'ADK',b'ADQ',b'AEX',b'AGS',b'ALB',b'AMA',b'ANC',b'ASE',b'ATL',b'ATW',b'AUS',b'AVL',b'AVP',b'AZO',b'BDL',b'BET',b'BFL',b'BGM',b'BGR',b'BHM',b'BIL',b'BIS',
                      b'BKG',b'BLI',b'BMI',b'BNA',b'BOI',b'BOS',b'BQK',b'BQN',b'BRO',b'BRW',b'BTM',b'BTR',b'BTV',b'BUF',b'BUR',b'BWI',b'BZN',b'CAE',b'CAK',b'CDC',b'CDV',b'CEC',b'CHA',b'CHO',b'CHS',b'CIC',b'CID',b'CLD',b'CLE',b'CLL',
                      b'CLT',b'CMH',b'CMI',b'CMX',b'COD',b'COS',b'COU',b'CPR',b'CRP',b'CRW',b'CSG',b'CVG',b'CWA',b'CYS',b'DAB',b'DAL',b'DAY',b'DBQ',b'DCA',b'DEN',b'DFW',b'DHN',b'DLH',b'DRO',b'DSM',b'DTW',b'EAU',b'ECP',b'EGE',b'EKO',
                      b'ELM',b'ELP',b'ERI',b'EUG',b'EVV',b'EWN',b'EWR',b'EYW',b'FAI',b'FAR',b'FAT',b'FAY',b'FCA',b'FLG',b'FLL',b'FLO',b'FNT',b'FSD',b'FSM',b'FWA',b'GCC',b'GEG',b'GFK',b'GGG',b'GJT',b'GNV',b'GPT',b'GRB',b'GRK',b'GRR',
                      b'GSO',b'GSP',b'GTF',b'GTR',b'GUC',b'GUM',b'HDN',b'HLN',b'HNL',b'HOU',b'HPN',b'HRL',b'HSV',b'HTS',b'IAD',b'IAH',b'ICT',b'IDA',b'ILM',b'IND',b'IPL',b'ISP',b'ITH',b'ITO',b'IYK',b'JAC',b'JAN',b'JAX',b'JFK',b'JNU',
                      b'KOA',b'KTN',b'LAN',b'LAS',b'LAX',b'LBB',b'LCH',b'LEX',b'LFT',b'LGA',b'LGB',b'LIH',b'LIT',b'LMT',b'LNK',b'LRD',b'LSE',b'LWB',b'LWS',b'LYH',b'MAF',b'MBS',b'MCI',b'MCO',b'MDT',b'MDW',b'MEI',b'MEM',b'MFE',b'MFR',
                      b'MGM',b'MHK',b'MHT',b'MIA',b'MKE',b'MKG',b'MLB',b'MLI',b'MLU',b'MMH',b'MOB',b'MOD',b'MOT',b'MQT',b'MRY',b'MSN',b'MSO',b'MSP',b'MSY',b'MTJ',b'MYR',b'OAJ',b'OAK',b'OGG',b'OKC',b'OMA',b'OME',b'ONT',b'ORD',b'ORF',
                      b'OTH',b'OTZ',b'PAH',b'PBI',b'PDX',b'PHF',b'PHL',b'PHX',b'PIA',b'PIE',b'PIH',b'PIT',b'PLN',b'PNS',b'PSC',b'PSE',b'PSG',b'PSP',b'PVD',b'PWM',b'RAP',b'RDD',b'RDM',b'RDU',b'RIC',b'RKS',b'RNO',b'ROA',b'ROC',b'ROW',
                      b'RST',b'RSW',b'SAF',b'SAN',b'SAT',b'SAV',b'SBA',b'SBN',b'SBP',b'SCC',b'SCE',b'SDF',b'SEA',b'SFO',b'SGF',b'SGU',b'SHV',b'SIT',b'SJC',b'SJT',b'SJU',b'SLC',b'SMF',b'SMX',b'SNA',b'SPI',b'SPS',b'SRQ',b'STL',b'STT',
                      b'STX',b'SUN',b'SWF',b'SYR',b'TEX',b'TLH',b'TOL',b'TPA',b'TRI',b'TUL',b'TUS',b'TVC',b'TWF',b'TXK',b'TYR',b'TYS',b'UTM',b'VLD',b'VPS',b'WRG',b'XNA',b'YAK',b'YUM'}
    airportFromList = []
    for s in airportFromSet:
        airportFromList.append(s)

    # replace_func = lambda x: 'bad.' if x in values_to_replace else x
    df['AirportFrom'] = df['AirportFrom'].apply(lambda x: airportFromList.index(x))

    airportToSet = {b'ABE',b'ABI',b'ABQ',b'ABR',b'ABY',b'ACT',b'ACV',b'ACY',b'ADK',b'ADQ',b'AEX',b'AGS',b'ALB',b'AMA',b'ANC',b'ASE',b'ATL',b'ATW',b'AUS',b'AVL',b'AVP',b'AZO',b'BDL',b'BET',b'BFL',b'BGM',b'BGR',b'BHM',b'BIL',b'BIS',
                      b'BKG',b'BLI',b'BMI',b'BNA',b'BOI',b'BOS',b'BQK',b'BQN',b'BRO',b'BRW',b'BTM',b'BTR',b'BTV',b'BUF',b'BUR',b'BWI',b'BZN',b'CAE',b'CAK',b'CDC',b'CDV',b'CEC',b'CHA',b'CHO',b'CHS',b'CIC',b'CID',b'CLD',b'CLE',b'CLL',
                      b'CLT',b'CMH',b'CMI',b'CMX',b'COD',b'COS',b'COU',b'CPR',b'CRP',b'CRW',b'CSG',b'CVG',b'CWA',b'CYS',b'DAB',b'DAL',b'DAY',b'DBQ',b'DCA',b'DEN',b'DFW',b'DHN',b'DLH',b'DRO',b'DSM',b'DTW',b'EAU',b'ECP',b'EGE',b'EKO',
                      b'ELM',b'ELP',b'ERI',b'EUG',b'EVV',b'EWN',b'EWR',b'EYW',b'FAI',b'FAR',b'FAT',b'FAY',b'FCA',b'FLG',b'FLL',b'FLO',b'FNT',b'FSD',b'FSM',b'FWA',b'GCC',b'GEG',b'GFK',b'GGG',b'GJT',b'GNV',b'GPT',b'GRB',b'GRK',b'GRR',
                      b'GSO',b'GSP',b'GTF',b'GTR',b'GUC',b'GUM',b'HDN',b'HLN',b'HNL',b'HOU',b'HPN',b'HRL',b'HSV',b'HTS',b'IAD',b'IAH',b'ICT',b'IDA',b'ILM',b'IND',b'IPL',b'ISP',b'ITH',b'ITO',b'IYK',b'JAC',b'JAN',b'JAX',b'JFK',b'JNU',
                      b'KOA',b'KTN',b'LAN',b'LAS',b'LAX',b'LBB',b'LCH',b'LEX',b'LFT',b'LGA',b'LGB',b'LIH',b'LIT',b'LMT',b'LNK',b'LRD',b'LSE',b'LWB',b'LWS',b'LYH',b'MAF',b'MBS',b'MCI',b'MCO',b'MDT',b'MDW',b'MEI',b'MEM',b'MFE',b'MFR',
                      b'MGM',b'MHK',b'MHT',b'MIA',b'MKE',b'MKG',b'MLB',b'MLI',b'MLU',b'MMH',b'MOB',b'MOD',b'MOT',b'MQT',b'MRY',b'MSN',b'MSO',b'MSP',b'MSY',b'MTJ',b'MYR',b'OAJ',b'OAK',b'OGG',b'OKC',b'OMA',b'OME',b'ONT',b'ORD',b'ORF',
                      b'OTH',b'OTZ',b'PAH',b'PBI',b'PDX',b'PHF',b'PHL',b'PHX',b'PIA',b'PIE',b'PIH',b'PIT',b'PLN',b'PNS',b'PSC',b'PSE',b'PSG',b'PSP',b'PVD',b'PWM',b'RAP',b'RDD',b'RDM',b'RDU',b'RIC',b'RKS',b'RNO',b'ROA',b'ROC',b'ROW',
                      b'RST',b'RSW',b'SAF',b'SAN',b'SAT',b'SAV',b'SBA',b'SBN',b'SBP',b'SCC',b'SCE',b'SDF',b'SEA',b'SFO',b'SGF',b'SGU',b'SHV',b'SIT',b'SJC',b'SJT',b'SJU',b'SLC',b'SMF',b'SMX',b'SNA',b'SPI',b'SPS',b'SRQ',b'STL',b'STT',
                      b'STX',b'SUN',b'SWF',b'SYR',b'TEX',b'TLH',b'TOL',b'TPA',b'TRI',b'TUL',b'TUS',b'TVC',b'TWF',b'TXK',b'TYR',b'TYS',b'UTM',b'VLD',b'VPS',b'WRG',b'XNA',b'YAK',b'YUM'}
    airportToList = []
    for s in airportToSet:
        airportToList.append(s)

    # replace_func = lambda x: 'bad.' if x in values_to_replace else x 
    df['AirportTo'] = df['AirportTo'].apply(lambda x: airportToList.index(x))
    # Display the DataFrame
    return {"airlines": df}


def getCovtype():
    data = arff.loadarff(base_path + 'covtype-normalized.arff')
    # Convert the ARFF data to a DataFrame
    df = pd.DataFrame(data[0])

    values_to_replace = {b'1', b'2', b'3', b'4', b'5', b'6', b'7'}

    labelList = []
    for s in values_to_replace:
        labelList.append(s)
    df['target'] = df['target'].apply(lambda x: int(x.decode()))

    col_list = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology',
                'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
                'Horizontal_Distance_To_Fire_Points', 'target']
    df = df[col_list]
    # Display the DataFrame
    return {"covtype": df}

def getGas():
    data = arff.loadarff(base_path + 'gas.arff')
    # Convert the ARFF data to a DataFrame
    df = pd.DataFrame(data[0])

    df['target'] = df['Class'].apply(lambda x: int(x.decode()))

    #col_list = ['date', 'day', 'period', 'nswprice', 'nswdemand','vicprice', 'vicdemand', 'transfer', 'target']
    col_list = ["V{}".format(i) for i in range(1, 129)]
    col_list.append("target")
    print(col_list)
    df = df[col_list]
    # Display the DataFrame
    return {"Gas": df}

def getAbruptInsects():
    with open(base_path + 'datasets_sample.pkl', 'rb') as fp:
        pd.options.display.float_format = '{:.2f}'.format
        np.set_printoptions(suppress=True)
        datasets = pickle.load(fp)
        datasets = datasets['AbruptInsects']
    return datasets

def getElec():
    data = arff.loadarff(base_path + 'electricity-normalized.arff')
    # Convert the ARFF data to a DataFrame
    df = pd.DataFrame(data[0])

    df['day'] = df['day'].apply(lambda x: int(x.decode()))


    values_to_replace = {b'UP', b'DOWN'}
    labelList = []
    for s in values_to_replace:
        labelList.append(s)
    df['target'] = df['class'].apply(lambda x: labelList.index(x))

    col_list = ['date', 'day', 'period', 'nswprice', 'nswdemand','vicprice', 'vicdemand', 'transfer', 'target']
    df = df[col_list]
    # Display the DataFrame
    return {"Electricity": df}

