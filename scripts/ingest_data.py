import pandas as pd

def load_synthea_data():
    patients = pd.read_csv('data/patients.csv')
    conditions = pd.read_csv('data/conditions.csv')

    patient_data = []
    for patient_id in patients['Id'].unique():
        patient_info = patients[patients['Id'] == patient_id].iloc[0]
        patient_conditions = conditions[conditions['PATIENT'] == patient_id]

        profile = f"Patient ID: {patient_id}\n"
        profile += f"Name: {patient_info['FIRST']} {patient_info['LAST']}\n"
        profile += f"Gender: {patient_info['GENDER']}, Birthdate: {patient_info['BIRTHDATE']}\n"
        profile += f"Conditions:\n"
        for _, row in patient_conditions.iterrows():
            #profile += f" - {row['DESCRIPTION']}\n"
            start = row.get('START', 'N/A')
            stop = row.get('STOP', 'N/A')
            profile += f" - {row['DESCRIPTION']} (Start: {start}, Stop: {stop})\n"

        patient_data.append((patient_id, profile))

    import pickle
    with open('agent/patient_records.pkl', 'wb') as f:
        pickle.dump(patient_data, f)

    return patient_data

# Execute this function
patientdata = load_synthea_data()
print(patientdata[0])
