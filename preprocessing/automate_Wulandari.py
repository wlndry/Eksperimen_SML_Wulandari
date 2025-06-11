# automate_NamaSiswa.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

def preprocess_data(df):
    # 1. Menghapus Missing Values
    if df.isnull().sum().sum() > 0:
        df = df.dropna()

    # 2. Menghapus Duplikat
    df = df.drop_duplicates()

    # 3. Deteksi dan Penanganan Outlier
    def remove_outliers_iqr(df, columns):
        for col in columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            df = df[(df[col] >= lower) & (df[col] <= upper)]
        return df

    numeric_cols = ['Age', 'ExperienceYears', 'PreviousCompanies', 'DistanceFromCompany',
                    'InterviewScore', 'SkillScore', 'PersonalityScore']
    df = remove_outliers_iqr(df, numeric_cols)

    # 4. Binning
    df['AgeGroup'] = pd.cut(df['Age'], bins=[20, 30, 40, 50, 60], labels=['20s', '30s', '40s', '50s'])
    df['ExperienceLevel'] = pd.cut(df['ExperienceYears'], bins=[0, 3, 7, 15, 30],
                                   labels=['Junior', 'Mid', 'Senior', 'Expert'])

    # 5. One-hot Encoding
    df = pd.get_dummies(df, columns=['AgeGroup', 'ExperienceLevel', 'RecruitmentStrategy'], drop_first=True)

    # 6. Standarisasi
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    return df

if __name__ == "__main__":

    # Ambil direktori tempat script ini berada
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Path ke file CSV mentah di folder utama
    input_path = os.path.join(base_dir, '..', 'recruitment_data_raw.csv')
    output_path = os.path.join(base_dir, 'recruitment_data_clean.csv')

    print("Path input:", input_path)
    print("Path output:", output_path)

    # Load dan proses
    df_raw = pd.read_csv(input_path)
    df_processed = preprocess_data(df_raw)

    # Simpan hasil
    df_processed.to_csv(output_path, index=False)
    print("Preprocessing selesai.")
    print(f"File disimpan di: {output_path}")
