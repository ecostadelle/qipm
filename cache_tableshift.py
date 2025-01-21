from tableshift import get_dataset
# f rom rfbs.ipm import AVALIABLE_DATASETS


AVALIABLE_DATASETS = [
    [ 'ASSISTments',             'assistments'             ], # 
    [ 'Childhood Lead',          'nhanes_lead'             ], # ok
    [ 'College Scorecard',       'college_scorecard'       ], # 
    [ 'Diabetes',                'brfss_diabetes'          ], # ok
    [ 'FICO HELOC',              'heloc'                   ], # ok
    [ 'Food Stamps',             'acsfoodstamps'           ], # ok
    [ 'Hospital Readmission',    'diabetes_readmission'    ], # ok
    [ 'Hypertension',            'brfss_blood_pressure'    ], # ok
    # [ 'ICU Length of Stay'     'mimic_extract_los_3'     ],
    # [ 'ICU Mortality',         'mimic_extract_mort_hosp' ],
    [ 'Income',                  'acsincome'               ], # 
    # [ 'Public Health Insurance', 'acspubcov'             ],
    [ 'Sepsis',                  'physionet'               ], # 
    [ 'Unemployment',            'acsunemployment'         ], # 
    [ 'Voting',                  'anes'                    ] # ok
    ]

for d,e in AVALIABLE_DATASETS:
    dset = get_dataset(e, "../tableshift/tmp", initialize_data=False)
    if not dset.is_cached():
        print(f"Downloading: {d}")
        dset._initialize_data()
        dset.to_sharded(domains_to_subdirectories=True)
