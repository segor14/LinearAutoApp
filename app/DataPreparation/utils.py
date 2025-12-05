import pandas as pd
import re
import numpy as np

#==== MODEL 1 ====

# ---- Справочники ----
FUEL_KEYWORDS = {
    'diesel': 'Diesel',
    'petrol': 'Petrol',
    'cng': 'CNG',
    'lpg': 'LPG',
    'hybrid': 'Hybrid',
    'electric': 'Electric',
    'ev': 'Electric',
}

BODY_KEYWORDS = {
    'suv': 'SUV',
    'hatchback': 'Hatchback',
    'hatch': 'Hatchback',
    'sedan': 'Sedan',
    'muv': 'MUV',
    'mpv': 'MPV',
    'van': 'Van',
    'coupe': 'Coupe',
    'convertible': 'Convertible',
    'wagon': 'Wagon',
    'estate': 'Estate',
    'crossover': 'Crossover',
    'ecosport': 'SUV',
    'duster': 'SUV',
    'creta': 'SUV',
    'xuv': 'SUV',
}

TRANSMISSION_KEYWORDS = {
    'amt': 'AMT',
    'automatic': 'Automatic',
    'auto': 'Automatic',
    'at': 'AT',
    'mt': 'MT',
    'dsg': 'DSG',
    'cvt': 'CVT',
    'manual': 'Manual',
}

SPORT_KEYWORDS = [
    'sport', 'sportz', 'gt', 'gti', 'rs', 'amg', 'm sport', 'n line', 'r-line'
]

TRIM_KEYWORDS = [
    'lxi', 'vxi', 'zxi', 'ldi', 'vdi', 'zdi',
    'era', 'magna', 'asta', 'sportz', 'trend', 'ambiente',
    'titanium', 'highline', 'comfortline', 'style', 'active'
]

EMISSION_PATTERNS = [
    r'\bbs[-\s]?([0-9ivx]+)\b',   # BSIII / BS IV / BS4 / BS6
]

DRIVE_PATTERNS = [
    r'\b(2wd)\b',
    r'\b(4wd)\b',
    r'\b(4x4)\b',
    r'\b(4x2)\b',
]

SERIES_PATTERNS = [
    # i10, i20, i30...
    r'\b(i[0-9]{1,2})\b',
    # X1, X3, X5, Q3, Q7, A3, A4, A6...
    r'\b([A-Z][0-9]{1,2})\b',
    # "3 Series", "5 Series"
    r'\b([0-9]\s+Series)\b'
]

# ===== Вспомогательные функции =====

def find_fuel(name_lower: str):
    for k, v in FUEL_KEYWORDS.items():
        if k in name_lower:
            return v
    return None

def find_body_type(name_lower: str):
    for k, v in BODY_KEYWORDS.items():
        if k in name_lower:
            return v
    return None

def find_transmission(name_lower: str):
    tokens = re.findall(r'[A-Za-z0-9+]+', name_lower)
    for t in tokens:
        if t in TRANSMISSION_KEYWORDS:
            return TRANSMISSION_KEYWORDS[t]
    for k, v in TRANSMISSION_KEYWORDS.items():
        if k in name_lower:
            return v
    return None

def find_sport(name_lower: str):
    for kw in SPORT_KEYWORDS:
        if kw in name_lower:
            return 1
    return 0

def find_trim(name_lower: str):
    for kw in TRIM_KEYWORDS:
        if re.search(r'\b' + re.escape(kw) + r'\b', name_lower):
            return kw.upper()
    return None

def find_emission_norm(name_lower: str):
    for pat in EMISSION_PATTERNS:
        m = re.search(pat, name_lower)
        if m:
            val = m.group(1).upper()
            val = val.replace('III', '3').replace('IV', '4').replace('VI', '6')
            return 'BS' + re.sub(r'[^0-9]', '', val) if re.search(r'[0-9]', val) else 'BS' + val
    return None

def find_drive(name_lower: str):
    for pat in DRIVE_PATTERNS:
        m = re.search(pat, name_lower)
        if m:
            return m.group(1).upper()
    return None

def find_engine_displacement(name: str):
    m = re.search(r'(\d\.\d)\s*l?', name.lower())
    if m:
        liters = float(m.group(1))
        return int(round(liters * 1000))  # в куб. см
    m = re.search(r'(\d{3,4})\s*(cc|cm3|cc\.?)', name.lower())
    if m:
        return int(m.group(1))
    m = re.search(r'\b(\d\.\d)\b', name.lower())
    if m:
        liters = float(m.group(1))
        return int(round(liters * 1000))
    return None

def find_series(name: str):
    for pat in SERIES_PATTERNS:
        m = re.search(pat, name)
        if m:
            return m.group(1)
    return None

def parse_name(name: str):
    if not isinstance(name, str):
        return {
            'brand': None,
            'drive': None,
            'model': None,
            'fuel': None,
            'engine_displacement': None,
            'is_sport': 0,
            'transmission': None,
            'body_type': None,
            'trim': None,
            'emission_norm': None,
            'series': None,
        }

    name_stripped = name.strip()
    name_lower = name_stripped.lower()
    tokens = name_stripped.split()

    brand = tokens[0] if tokens else None
    rest = ' '.join(tokens[1:]) if len(tokens) > 1 else ''
    fuel = find_fuel(name_lower)
    engine_disp = find_engine_displacement(name_stripped)
    is_sport = find_sport(name_lower)
    transmission = find_transmission(name_lower)
    body_type = find_body_type(name_lower)
    trim = find_trim(name_lower)
    emission_norm = find_emission_norm(name_lower)
    drive = find_drive(name_lower)
    series = find_series(name_stripped)
    tech_words = set([
        'diesel', 'petrol', 'cng', 'lpg', 'hybrid', 'electric', 'ev',
        'amt', 'automatic', 'auto', 'at', 'mt', 'dsg', 'cvt', 'manual',
        'suv', 'sedan', 'hatchback', 'hatch', 'muv', 'mpv', 'van', 'coupe',
        'convertible', 'wagon', 'estate', 'crossover',
    ] + TRIM_KEYWORDS)

    model_tokens = []
    for t in rest.split():
        t_clean = re.sub(r'[^A-Za-z0-9]', '', t).lower()
        if t_clean in tech_words:
            continue
        if re.match(r'bs[0-9ivx]+', t_clean):
            continue
        if t_clean in ['4x4', '4x2', '2wd', '4wd']:
            continue
        model_tokens.append(t)

    model = ' '.join(model_tokens).strip() or None

    return {
        'brand': brand,
        'drive': drive,
        'model': model,
        'fuel': fuel,
        'engine_displacement': engine_disp,
        'is_sport': is_sport,
        'transmission': transmission,
        'body_type': body_type,
        'trim': trim,
        'emission_norm': emission_norm,
        'series': series,
    }

def parse_name_series(s: pd.Series) -> pd.DataFrame:
    parsed = s.apply(parse_name)
    return pd.DataFrame(list(parsed))


# ==== MODEL 2 ====
def split_x(x, format=float):
    try:
        return format(x)
    except:
        data = x.split(' ')[0]
        try:
            return format(data)
        except:
            return np.nan
    
def astype_numeric(df_orig, col_names, format):
    df = df_orig.copy()
    if isinstance(col_names, list):
        for col_name in col_names:
            df[col_name] = df[col_name].apply(split_x, format=format)
    else:
        df[col_names] = df[col_names].apply(split_x, format=format)

    return df

def split_torque(x):
    if isinstance(x, str):
        x = x.lower()
        if '+/-' in x:
            x_list = x.split('@ ')
            torque = x_list[0]
            if 'nm' in torque:
                torque = float(torque.split('nm')[0])
            max_torque = x_list[1].split('+/-')[0]
            max_torque = int(max_torque.replace(',', ''))
        elif 'nm@ ' in x:
            x_list = x.split('nm@ ')
            torque = float(x_list[0])
            max_torque = x_list[1][:-3]
            if '-' in max_torque:
                max_torque = max_torque.split('-')[-1]
            elif '~' in max_torque:
                max_torque = max_torque.split('~')[-1]
            max_torque = int(max_torque.replace(',', ''))
        elif '(kgm@ rpm)' in x:
            x_list = x.split('@ ')
            torque = float(x_list[0]) * 9.80665
            max_torque = x_list[1].split('(')[0]
            if '-' in max_torque:
                max_torque = max_torque.split('-')[-1]
            max_torque = int(max_torque.replace(',', ''))
        elif 'kgm@ ' in x:
            x_list = x.split('kgm@ ')
            torque = float(x_list[0]) * 9.80665
            max_torque = x_list[1][:-3]
            if '-' in max_torque:
                max_torque = int(max_torque.split('-')[-1])
        elif 'kgm at ' in x:
            x_list = x.split(' kgm at ')
            torque = float(x_list[0]) * 9.80665
            max_torque = x_list[1].split('-')[-1][:-3]
            max_torque = int(max_torque.replace(',', ''))
        elif 'nm at ' in x:
            x_list = x.split('nm at ')
            torque = float(x_list[0])
            max_torque = x_list[1].split('-')[-1].split('r')[0]
            max_torque = int(max_torque.replace(',', ''))
        elif '@ ' in x:
            x_list = x.split('@ ')
            torque = x_list[0]
            if 'nm' in torque:
                torque = torque.split('nm')[0]
            if '(' in torque:
                torque = torque.split('(')[0]
            torque = float(torque)
            max_torque = x_list[1][:-3]
            if '-' in max_torque:
                max_torque = int(max_torque.split('-')[-1])
        elif 'nm' in x:
            x_list = x.split('nm')
            torque = float(x_list[0])
            max_torque = np.nan
        elif ' / ' in x:
            x_list = x.split(' / ')
            torque = float(x_list[0])
            max_torque = float(x_list[1])
        else:
            torque, max_torque = np.nan, np.nan
    else:
        torque, max_torque = np.nan, np.nan

    return float(torque), float(max_torque)

def apply_split_torque(df_orig):
    df = df_orig.copy()
    torque_values, rpm_values = zip(*df['torque'].apply(split_torque))
    df['torque'] = torque_values
    df['max_torque_rpm'] = rpm_values

    return df

def create_new_features(df_orig):
    df = df_orig.copy()
    df['horse*volume'] = df['engine'] * df['max_power']
    df['year^2'] = df['year'] ** 2
    df = df.drop('year', axis=1)

    return df

def new_features(df_orig):
    df = df_orig.copy()
    df['FS_owner'] = df['owner'].isin(['First Owner', 'Second Owner']) * 1
    df['TF_owner'] = df['owner'].isin(['Third Owner', 'Fourth & Above Owner']) * 1
    df = df.drop('owner', axis=1)

    return df

def fill_outliers(df_orig, bounds):
    df = df_orig.copy()
    for col in df:
        lower_bound, upper_bound = bounds[col]
        df[col] = np.where(df[col] < lower_bound, lower_bound,
                              np.where(df[col] > upper_bound, upper_bound, 
                              df[col]))
        
        return df
    
def log_col(df_orig):
    df = df_orig.copy()
    for col in ['km_driven', 'engine', 'max_power', 'torque', 'max_torque_rpm', 'horse*volume', 'year^2']:
        df[col] = np.log(df[col]+1e-6)

    return df