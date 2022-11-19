MODEL_SETTING = {
    'PeMS20': {
        'day_interval': int(24 * 60 / 5)
    },
    'Beijing': {
        'day_interval': int(24 * 60 / 5)
    },
    'Electricity':{
        'day_interval': int(24 * 60 / 15)
    },
    'COVID-US':{
        'day_interval': int(24 * 60 / 60)
    },
    'COVID-CHI':{
        'day_interval': int(24 * 60 / 60) # setting to 24 to avoid overlap
    },
    'Japan-OD':{
        'day_interval': int(24 * 60 / 24 * 60)
    },
    'Japan-Density':{
        'day_interval': int(24 * 60 / 60)
    }
}